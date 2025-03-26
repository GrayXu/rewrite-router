const express = require('express');
const fetch = require('node-fetch');
const fs = require('fs');
const bodyParser = require('body-parser');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const { Readable } = require('stream');
const chalk = require('chalk');

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
  }
};

// Calculate the token count of messages
const getMessagesTokenCount = (messages) => {
  if (!Array.isArray(messages)) return 0;

  let totalTokens = 2; // Base format overhead
  const perMessageTokens = 4; // Role marker tokens

  for (const { content } of messages) {
    totalTokens += perMessageTokens;

    if (typeof content === 'string') {
      totalTokens += tokenizer.encode(content).length;
    } else if (Array.isArray(content)) {
      let combinedText = '';
      for (const part of content) {
        if (typeof part.text === 'string') {
          combinedText += part.text;
        }
      }
      totalTokens += tokenizer.encode(combinedText).length;
    }
  }

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
    // Modified: Use toLocaleString() to output local time
    console.log(`[${new Date().toLocaleString()}] Received chat completions request: ${req.headers['content-type']}`);

    // Ensure we have a JSON formatted request body
    let data;
    if (req.headers['content-type'] && req.headers['content-type'].includes('application/json')) {
      data = req.body;
    } else {
      try {
        data = JSON.parse(req.body.toString());
      } catch (e) {
        // Modified: Use toLocaleString() to output local time
        console.error(`[${new Date().toLocaleString()}] Failed to parse request body as JSON`);
        return res.status(400).json({ error: 'Invalid JSON in request body' });
      }
    }

    // Modified: Use chalk.red() to output the model name in red, and use toLocaleString() to output local time
    console.log(`[${new Date().toLocaleString()}] Request model: ${chalk.red(data.model)}`);

    // Handle automatic model routing selection
    if (ROUTING_RULES[data.model]) {
      const selectedModel = selectModel(data.messages, ROUTING_RULES[data.model]);
      if (selectedModel) {
        // Modified: Use chalk.red() to output the model name in red, and use toLocaleString() to output local time
        console.log(`[${new Date().toLocaleString()}] Auto-routing: ${chalk.red(data.model)} -> ${chalk.red(selectedModel)}`);
        data.model = selectedModel; // Update the model in data to the selected model
      }
    }

    // Apply rewrite rules
    if (REWRITE_RULES[data.model]) { // Use the updated data.model
      // Modified: Use chalk.red() to output the model name in red, and use toLocaleString() to output local time
      console.log(`[${new Date().toLocaleString()}] Applying rewrite rules for ${chalk.red(data.model)}`);
      const rules = REWRITE_RULES[data.model];

      Object.entries(rules).forEach(([key, value]) => {
        if (key === 'message') {
          data.messages = [...value, ...(data.messages || [])];
        } else if (key === 'tools') {
          data.tools = [...(data.tools || []), ...value];
        } else if (key === 'stream' && value === 'false') {
          data.stream = false;
        } else {
          // Modified: Use toLocaleString() to output local time
          console.log(`[${new Date().toLocaleString()}] \t\trewrite rule: ${key} = ${value}`);
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

    // Modified: Use toLocaleString() to output local time
    console.log(`[${new Date().toLocaleString()}] Forwarding data:`, data); // Print the final data sent to the backend

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
      // console.log(`[${new Date().toLocaleString()}] Streaming response back to client`); // Uncomment as needed
      const reader = response.body;
      reader.pipe(res);

      // Error handling
      reader.on('error', (err) => {
        // Modified: Use toLocaleString() to output local time
        console.error(`[${new Date().toLocaleString()}] Stream error:`, err);
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
    // Modified: Use toLocaleString() to output local time
    console.error(`[${new Date().toLocaleString()}] Error processing chat completions: ${error.stack}`);
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
    // console.log(`[${new Date().toLocaleString()}] Fetching models list from backend`); // Uncomment as needed
    const headers = { ...req.headers };
    delete headers['host'];

    const response = await fetch(`${BACKEND_URL}/v1/models`, { headers });

    if (!response.ok) {
      // Modified: Use toLocaleString() to output local time
      console.error(`[${new Date().toLocaleString()}] Backend returned error: ${response.status}`);
      return res.status(response.status).send(await response.text());
    }

    const responseData = await response.json();

    if (!Array.isArray(responseData?.data)) {
      // Modified: Use toLocaleString() to output local time
      console.error(`[${new Date().toLocaleString()}] Invalid response format from backend`);
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
    // Modified: Use toLocaleString() to output local time
    console.error(`[${new Date().toLocaleString()}] Error fetching models: ${e.stack}`);
    res.status(502).json({ error: { message: "Error connecting to backend", type: "proxy_error" } });
  }
});

// General proxy handling for other routes
// General proxy handling for other routes
app.all('*', async (req, res) => {
  try {
    const path = req.path;
    const method = req.method;
    const timestamp = new Date().toLocaleString(); // Store timestamp for consistent logging
    console.log(`[${timestamp}] Proxying ${method} request to: ${path}`);

    // Prepare request headers
    const headers = { ...req.headers };
    // Keep host header from original request unless it's the proxy's own host/port
    // Or let node-fetch determine the host based on targetUrl, which is usually better.
    // For simplicity, let's continue deleting it and let node-fetch handle it.
    delete headers['host'];
    // Content-Length will be automatically handled by node-fetch based on the body
    delete headers['content-length']; // Remove potentially incorrect length from original req

    // Prepare request options
    const fetchOptions = {
      method,
      headers,
      redirect: 'follow'
    };

    // Handle request body
    // Ensure req.body is correctly parsed before this middleware if needed
    // The express.raw middleware should provide req.body as a Buffer for non-text/json types
    if (['POST', 'PUT', 'PATCH'].includes(method) && req.body) {
       // Check if body is empty buffer or empty object/string before assigning
       if (Buffer.isBuffer(req.body) && req.body.length > 0) {
           fetchOptions.body = req.body;
           // Ensure content-type is passed correctly
           if (!headers['content-type']) {
               headers['content-type'] = req.headers['content-type'] || 'application/octet-stream'; // Default if unknown
           }
       } else if (typeof req.body === 'string' && req.body.length > 0) {
           fetchOptions.body = req.body;
           if (!headers['content-type']) {
                headers['content-type'] = req.headers['content-type'] || 'text/plain';
           }
       } else if (typeof req.body === 'object' && Object.keys(req.body).length > 0) {
            // Assuming JSON if it's an object and content-type was application/json
           if (req.is('application/json')) {
               fetchOptions.body = JSON.stringify(req.body);
               headers['content-type'] = 'application/json'; // Ensure correct header
           } else {
               // If it's an object but not json, it might be urlencoded form data
               // Body-parser middleware should ideally handle this before it gets here
               // For safety, let's log a warning if we get an object without json type
               console.warn(`[${timestamp}] Received object body for ${method} ${path} without application/json content-type. Sending as is.`);
               fetchOptions.body = req.body; // May or may not work depending on backend
           }
       }
    }


    // Send request to backend
    // Construct URL carefully: Use req.originalUrl to preserve the full path and query string
    const targetUrl = `${BACKEND_URL}${req.originalUrl}`;
    console.log(`[${timestamp}] Forwarding to: ${targetUrl}`);

    const response = await fetch(targetUrl, fetchOptions);
    console.log(`[${timestamp}] Backend response status: ${response.status}`);

    // Set response headers from backend response
    Object.entries(response.headers.raw()).forEach(([key, values]) => {
      // Avoid setting encoding/transfer headers that might conflict with Node's handling
      const lowerKey = key.toLowerCase();
      if (lowerKey !== 'content-encoding' && lowerKey !== 'transfer-encoding' && lowerKey !== 'connection') {
         // Note: 'connection' header is also often problematic for proxies
        res.set(key, values.join(', '));
      }
    });

    // Set status code
    res.status(response.status);

    // --- MODIFICATION START ---
    // Handle response body: Read fully for non-streaming, pipe for known streaming types if needed

    const contentType = response.headers.get('content-type');

    // Define types you might want to explicitly stream (if any passed through here)
    const streamingContentTypes = ['text/event-stream']; // Add other types like video/audio if needed

    // Decide whether to stream or buffer
    // Simple approach: buffer everything unless it's a known streaming type
    if (response.body && contentType && streamingContentTypes.some(type => contentType.includes(type))) {
        console.log(`[${timestamp}] Piping streaming response for Content-Type: ${contentType}`);
        response.body.pipe(res);
        response.body.on('error', (pipeErr) => {
            console.error(`[${timestamp}] Error piping backend stream:`, pipeErr);
            if (!res.headersSent) {
                 res.status(500).send('Proxy stream error');
            } else {
                 res.end(); // Attempt to close the connection cleanly
            }
        });
    } else if (response.body) {
        // For non-streaming or unknown types, read the full body first
        try {
            // console.log(`[${timestamp}] Buffering response body for Content-Type: ${contentType || 'N/A'}`);
            const bodyBuffer = await response.buffer(); // Read body into a buffer
            res.send(bodyBuffer); // Send the buffer
            // console.log(`[${timestamp}] Successfully sent buffered response.`);
        } catch (bufferError) {
            console.error(`[${timestamp}] Error reading backend response body:`, bufferError);
            if (!res.headersSent) {
               res.status(500).send('Error reading backend response');
            } else {
               res.end(); // End response if headers already sent
            }
        }
    } else {
        // No response body (e.g., for 204 No Content)
        res.end();
    }
    // --- MODIFICATION END ---

  } catch (error) {
    const timestamp = new Date().toLocaleString();
    console.error(`[${timestamp}] Proxy error: ${error.stack}`);
    if (!res.headersSent) {
      res.status(502).json({ // Send JSON error for consistency? Or text?
        error: {
          message: 'Error proxying request to backend',
          type: 'proxy_error',
          details: error.message // Include underlying error message
        }
      });
    } else {
        // If headers were sent (e.g., during streaming), just end the response.
        res.end();
    }
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  // Modified: Use toLocaleString() to output local time
  console.error(`[${new Date().toLocaleString()}] Express error:`, err.stack);
  if (!res.headersSent) { // Check if headers have been sent
    res.status(500).json({
      error: {
        message: 'Internal server error',
        type: 'server_error'
      }
    });
  } else {
    // If headers have been sent (e.g., during streaming), try to end the response
    res.end();
  }
});

// Start the server
const argv = yargs(hideBin(process.argv))
  .option('host', { type: 'string', default: '127.0.0.1' })
  .option('port', { type: 'number', default: 3034 })
  .parse();

const server = app.listen(argv.port, argv.host, () => {
  // Modified: Use toLocaleString() to output local time
  console.log(`[${new Date().toLocaleString()}] Server running at http://${argv.host}:${argv.port}`);
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
  // Modified: Use toLocaleString() to output local time
  console.log(`[${new Date().toLocaleString()}] Server shutting down, resources freed`);
});

process.on('SIGINT', () => {
  // Modified: Use toLocaleString() to output local time
  console.log(`[${new Date().toLocaleString()}] Received SIGINT, shutting down`);
  server.close(() => {
    process.exit(0);
  });
});

process.on('SIGTERM', () => {
  // Modified: Use toLocaleString() to output local time
  console.log(`[${new Date().toLocaleString()}] Received SIGTERM, shutting down`);
  server.close(() => {
    process.exit(0);
  });
});

// Handle uncaught exceptions and rejections
process.on('uncaughtException', (err) => {
  // Modified: Use toLocaleString() to output local time
  console.error(`[${new Date().toLocaleString()}] Uncaught exception:`, err);
  // Consider gracefully closing the server here, or letting the process manager handle restarts
  // process.exit(1); // Forcing exit may interrupt requests
});

process.on('unhandledRejection', (reason, promise) => {
  // Modified: Use toLocaleString() to output local time
  console.error(`[${new Date().toLocaleString()}] Unhandled Rejection at:`, promise, 'reason:', reason);
  // Similarly, consider whether to exit or log more details
});