const express = require('express');
const fetch = require('node-fetch');
const fs = require('fs');
const { createParser } = require('eventsource-parser'); // 实际上在这个版本里没用到，但为了完整性保留
const bodyParser = require('body-parser');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');

// Tiktoken 相关引入和初始化 (修复部分)
const { Tiktoken } = require("@dqbd/tiktoken");
const cl100k_base = require("@dqbd/tiktoken/encoders/cl100k_base.json");

const tokenizer = new Tiktoken(
  cl100k_base.bpe_ranks,
  cl100k_base.special_tokens,
  cl100k_base.pat_str
);

// 初始化配置
const config = JSON.parse(fs.readFileSync('config.json', 'utf-8'));
const BACKEND_URL = config.BACKEND_URL;
const REWRITE_RULES = config.REWRITE_RULES;
const ROUTING_RULES = config.ROUTING_RULES;

const app = express();
app.use(bodyParser.json());
app.use(bodyParser.text({ type: 'text/plain' }));

// 公共工具函数 (修复部分)
const getTokenCount = (text) => {
  try {
    return tokenizer.encode(text).length;
  } finally {
    tokenizer.free();  // 重要：每次使用后释放
  }
};

const selectModel = (prompt, modelRouting) => {
  const tokenCount = getTokenCount(prompt);
  const contextLengths = Object.keys(modelRouting.models)
    .map(Number)
    .sort((a, b) => a - b);

  for (const length of contextLengths) {
    if (tokenCount <= length * modelRouting.threshold) {
      return modelRouting.models[length];
    }
  }
  return modelRouting.models[Math.max(...contextLengths)];
};

// 处理聊天完成接口
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const data = req.body;

    if (!data?.messages || !Array.isArray(data.messages)) {
      return res.status(400).send("Invalid request: 'messages' field is missing or not a list");
    }

    if (!data.model) {
      return res.status(400).send("Invalid request: 'model' field is missing");
    }

    // 模型路由
    if (ROUTING_RULES[data.model]) {
      const prompt = data.messages.map(m => m.content || '').join('');
      const modelSelected = selectModel(prompt, ROUTING_RULES[data.model]);
      console.log(`\t\trouting ${data.model} to ${modelSelected}`);
      data.model = modelSelected;
    }

    // 请求重写
    if (REWRITE_RULES[data.model]) {
      console.log(`\t\trewrite ${data.model}`);
      const rules = REWRITE_RULES[data.model];

      Object.entries(rules).forEach(([key, value]) => {
        if (key === 'message') {
          data.messages.unshift(...value);
        } else if (key === 'tools') {
          data.tools = [...(data.tools || []), ...value];
        } else {
          data[key] = value;
        }
      });
    }

    // 转发请求
    const headers = { ...req.headers };
    delete headers['host'];

    const response = await fetch(`${BACKEND_URL}/v1/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(data),
    });

    // 处理流式响应
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    response.body.pipe(res);
  } catch (e) {
    console.error(`Error forwarding chat completion: ${e}`);
    res.status(502).send(`Error forwarding request: ${e}`);
  }
});

// 处理模型列表接口
app.get('/v1/models', async (req, res) => {
  try {
    const headers = { ...req.headers };
    delete headers['host'];

    const response = await fetch(`${BACKEND_URL}/v1/models`, { headers });
    const responseData = await response.json();

    if (!Array.isArray(responseData?.data)) {
      return res.status(500).send("Invalid response from backend");
    }

    // 合并模型列表
    const modelList = [...responseData.data];
    const created = modelList[0]?.created || 1677649963;

    Object.keys(REWRITE_RULES).forEach(modelName => {
      if (!modelList.some(m => m.id === modelName)) {
        modelList.push({
          id: modelName,
          object: "model",
          created,
          owned_by: "user"
        });
      }
    });

    res.json({ ...responseData, data: modelList });
  } catch (e) {
    console.error(`Error fetching models: ${e}`);
    res.status(502).send("Error connecting to backend");
  }
});

// 通用代理处理
app.all('*', async (req, res) => {
  try {
    const path = req.params[0];
    const url = new URL(path, BACKEND_URL);

    // 保留查询参数
    Array.from(req.query.entries()).forEach(([key, value]) => {
      url.searchParams.append(key, value);
    });

    const headers = { ...req.headers };
    delete headers['host'];

    const fetchOptions = {
      method: req.method,
      headers,
      body: req.method === 'GET' ? null : req,
    };

    const response = await fetch(url.toString(), fetchOptions);

    // 复制响应头
    Array.from(response.headers.entries()).forEach(([key, value]) => {
      res.setHeader(key, value);
    });

    res.status(response.status);
    response.body.pipe(res);
  } catch (e) {
    console.error(`Proxy error: ${e}`);
    res.status(502).send("Error forwarding request");
  }
});

// 启动服务
const argv = yargs(hideBin(process.argv))
  .option('host', { type: 'string', default: '127.0.0.1' })
  .option('port', { type: 'number', default: 3034 })
  .parse();

app.listen(argv.port, argv.host, () => {
  console.log(`Server running at http://${argv.host}:${argv.port}`);
});

// 确保程序退出时释放资源 (修复部分)
process.on('exit', () => {
  tokenizer.free();
});

process.on('SIGINT', () => {
  process.exit();  // 显式退出，触发 'exit' 事件
});

process.on('SIGTERM', () => {
   process.exit();
});