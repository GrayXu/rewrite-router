const express = require('express');
const fetch = require('node-fetch');
const fs = require('fs');
const bodyParser = require('body-parser');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const { Readable } = require('stream');

// Tiktoken related imports and initialization
const { Tiktoken } = require("@dqbd/tiktoken");
const cl100k_base = require("@dqbd/tiktoken/encoders/cl100k_base.json");

const tokenizer = new Tiktoken(
  cl100k_base.bpe_ranks,
  cl100k_base.special_tokens,
  cl100k_base.pat_str
);

// Initialize configuration
let config;
try {
  config = JSON.parse(fs.readFileSync('config.json', 'utf-8'));
} catch (error) {
  console.error('Failed to load config file:', error);
  process.exit(1);
}

const BACKEND_URL = config.BACKEND_URL;
const REWRITE_RULES = config.REWRITE_RULES || {};
const ROUTING_RULES = config.ROUTING_RULES || {};

const app = express();
// Increase request body size limit
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.text({ type: 'text/plain', limit: '50mb' }));
// Add raw body parsing to handle streaming requests
app.use(express.raw({ type: '*/*', limit: '50mb' }));

// Common utility function
const getTokenCount = (text) => {
  if (typeof text !== 'string') {
    return 0;
  }
  
  try {
    const tokens = tokenizer.encode(text);
    return tokens.length;
  } catch (error) {
    console.error('Error counting tokens:', error);
    return 0;
  } finally {
    tokenizer.free();
  }
};

// Calculate the token count of messages
const getMessagesTokenCount = (messages) => {
  if (!Array.isArray(messages)) {
    return 0;
  }
  
  let totalTokens = 0;
  
  for (const message of messages) {
    if (typeof message.content === 'string') {
      totalTokens += getTokenCount(message.content);
    } else if (Array.isArray(message.content)) {
      // Handle multimodal content
      for (const part of message.content) {
        if (typeof part.text === 'string') {
          totalTokens += getTokenCount(part.text);
        }
      }
    }
    
    // Add tokens for role markers (approximate)
    totalTokens += 4; // Base overhead per message
  }
  
  // Add base format overhead
  totalTokens += 2;
  
  return totalTokens;
};

const selectModel = (messages, modelRouting) => {
  if (!modelRouting || !modelRouting.models || !modelRouting.threshold) {
    console.warn('Invalid model routing configuration');
    return null;
  }
  
  // Convert messages to a format that can calculate tokens
  let promptText = '';
  if (Array.isArray(messages)) {
    const tokenCount = getMessagesTokenCount(messages);
    
    const contextLengths = Object.keys(modelRouting.models)
      .map(Number)
      .sort((a, b) => a - b);

    for (const length of contextLengths) {
      if (tokenCount <= length * modelRouting.threshold) {
        return modelRouting.models[length];
      }
    }
    return modelRouting.models[Math.max(...contextLengths)];
  } else {
    console.warn('Messages is not an array, cannot route model');
    return null;
  }
};

// Handle /v1/chat/completions route
app.post('/v1/chat/completions', async (req, res) => {
  try {
    console.log(`Received chat completions request: ${req.headers['content-type']}`);
    
    // Ensure we have a JSON formatted request body
    let data;
    if (req.headers['content-type'] && req.headers['content-type'].includes('application/json')) {
      data = req.body;
    } else {
      try {
        data = JSON.parse(req.body.toString());
      } catch (e) {
        console.error('Failed to parse request body as JSON');
        return res.status(400).json({ error: 'Invalid JSON in request body' });
      }
    }
    
    console.log(`Request model: ${data.model}`);
    
    // Handle automatic model routing selection
    if (ROUTING_RULES[data.model]) {
      const selectedModel = selectModel(data.messages, ROUTING_RULES[data.model]);
      if (selectedModel) {
        console.log(`Auto-routing: ${data.model} -> ${selectedModel}`);
        data.model = selectedModel;
      }
    }
    
    // Apply rewrite rules
    if (REWRITE_RULES[data.model]) {
      console.log(`Applying rewrite rules for ${data.model}`);
      const rules = REWRITE_RULES[data.model];

      Object.entries(rules).forEach(([key, value]) => {
        if (key === 'message') {
          data.messages = [...value, ...(data.messages || [])];
        } else if (key === 'tools') {
          data.tools = [...(data.tools || []), ...value];
        } else if (key === 'stream' && value === 'false') {
          data.stream = false;
        } else {
          data[key] = value;
        }
      });
    }

    // Forward request to backend
    const headers = { ...req.headers };
    delete headers['host'];
    delete headers['content-length'];
    headers['content-type'] = 'application/json';

    const fetchOptions = {
      method: 'POST',
      headers,
      body: JSON.stringify(data)
    };

    console.log(`Forwarding request to ${BACKEND_URL}/v1/chat/completions`);

    const response = await fetch(`${BACKEND_URL}/v1/chat/completions`, fetchOptions);
    
    // Set corresponding response headers
    Object.entries(response.headers.raw()).forEach(([key, values]) => {
      if (key.toLowerCase() !== 'content-encoding' && key.toLowerCase() !== 'transfer-encoding') {
        res.set(key, values.join(', '));
      }
    });
    
    res.status(response.status);
    
    // Handle streaming response
    if (data.stream) {
      // console.log('Streaming response back to client');
      const reader = response.body;
      reader.pipe(res);
      
      // Error handling
      reader.on('error', (err) => {
        console.error('Stream error:', err);
        if (!res.headersSent) {
          res.status(500).json({ error: 'Stream processing error' });
        } else {
          res.end();
        }
      });
    } else {
      // Non-streaming response
      const responseData = await response.text();
      res.send(responseData);
    }
  } catch (error) {
    console.error(`Error processing chat completions: ${error.stack}`);
    if (!res.headersSent) {
      res.status(502).json({ 
        error: { 
          message: 'Error forwarding request to backend', 
          type: 'proxy_error' 
        } 
      });
    }
  }
});

// Handle model list endpoint
app.get('/v1/models', async (req, res) => {
  try {
    // console.log('Fetching models list from backend');
    const headers = { ...req.headers };
    delete headers['host'];

    const response = await fetch(`${BACKEND_URL}/v1/models`, { headers });
    
    if (!response.ok) {
      console.error(`Backend returned error: ${response.status}`);
      return res.status(response.status).send(await response.text());
    }
    
    const responseData = await response.json();

    if (!Array.isArray(responseData?.data)) {
      console.error('Invalid response format from backend');
      return res.status(500).json({ error: "Invalid response from backend" });
    }

    // Merge model list
    const modelList = [...responseData.data];
    const created = modelList[0]?.created || 1677649963;

    // Add models from routing rules
    Object.keys(ROUTING_RULES).forEach(modelName => {
      if (!modelList.some(m => m.id === modelName)) {
        modelList.push({
          id: modelName,
          object: "model",
          created,
          owned_by: "routing"
        });
      }
    });

    // Add models from rewrite rules
    Object.keys(REWRITE_RULES).forEach(modelName => {
      if (!modelList.some(m => m.id === modelName)) {
        modelList.push({
          id: modelName,
          object: "model",
          created,
          owned_by: "rewrite"
        });
      }
    });

    res.json({ ...responseData, data: modelList });
  } catch (e) {
    console.error(`Error fetching models: ${e.stack}`);
    res.status(502).json({ error: { message: "Error connecting to backend", type: "proxy_error" } });
  }
});

// General proxy handling for other routes
app.all('*', async (req, res) => {
  try {
    const path = req.path;
    const method = req.method;
    console.log(`Proxying ${method} request to: ${path}`);

    // Prepare request headers
    const headers = { ...req.headers };
    delete headers['host'];
    
    // Prepare request options
    const fetchOptions = {
      method,
      headers,
      redirect: 'follow'
    };

    // Handle request body
    if (['POST', 'PUT', 'PATCH'].includes(method)) {
      if (req.is('application/json')) {
        fetchOptions.body = JSON.stringify(req.body);
        headers['content-type'] = 'application/json';
      } else if (req.is('text/*')) {
        fetchOptions.body = req.body;
      } else {
        // Raw request body
        fetchOptions.body = req.body;
      }
    }

    // Send request to backend
    const targetUrl = `${BACKEND_URL}${path}${req.url.replace(req.path, '')}`;
    console.log(`Forwarding to: ${targetUrl}`);
    
    const response = await fetch(targetUrl, fetchOptions);
    
    // Set response headers
    Object.entries(response.headers.raw()).forEach(([key, values]) => {
      if (key.toLowerCase() !== 'content-encoding' && key.toLowerCase() !== 'transfer-encoding') {
        res.set(key, values.join(', '));
      }
    });
    
    // Set status code
    res.status(response.status);
    
    // Forward response body
    const contentType = response.headers.get('content-type');
    
    // If it's a streaming response, stream directly
    if (response.body) {
      response.body.pipe(res);
    } else {
      res.end();
    }

  } catch (error) {
    console.error(`Proxy error: ${error.stack}`);
    if (!res.headersSent) {
      res.status(502).json({ 
        error: { 
          message: 'Error proxying request to backend', 
          type: 'proxy_error' 
        } 
      });
    }
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Express error:', err.stack);
  res.status(500).json({
    error: {
      message: 'Internal server error',
      type: 'server_error'
    }
  });
});

// Start the server
const argv = yargs(hideBin(process.argv))
  .option('host', { type: 'string', default: '127.0.0.1' })
  .option('port', { type: 'number', default: 3034 })
  .parse();

const server = app.listen(argv.port, argv.host, () => {
  console.log(`Server running at http://${argv.host}:${argv.port}`);
});

// Set timeout to 5 minutes to handle long requests
server.timeout = 5 * 60 * 1000;

// Ensure resource cleanup
process.on('exit', () => {
  try {
    tokenizer.free();
  } catch (e) {
    // Ignore possible errors
  }
  console.log('Server shutting down, resources freed');
});

process.on('SIGINT', () => {
  console.log('Received SIGINT, shutting down');
  server.close(() => {
    process.exit(0);
  });
});

process.on('SIGTERM', () => {
  console.log('Received SIGTERM, shutting down');
  server.close(() => {
    process.exit(0);
  });
});

// Handle uncaught exceptions and rejections
process.on('uncaughtException', (err) => {
  console.error('Uncaught exception:', err);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});