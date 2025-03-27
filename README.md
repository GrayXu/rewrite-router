# Rewrite router

This is an OpenAI-compatible API router designed to:

- **Rewrite request parameters**: Adjust parameters sent in API requests.
- **Add system prompts**: Add system prompts to guide the model's behavior.
- **Add tools**: Add tools, such as enabling Gemini's grounding search.
- **Route requests based on context token length**: Similar to moonshot-v1-auto, it intelligently routes requests to different model endpoints (e.g., `moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k`) depending on the input's token count.
This proxy is suited for deployment in front of intermediary platforms like one-api, one-hub, or new-api.

```bash
npm install
# Edit `config.json` to match your specific needs.
vim config.json
node rewrite.js --host-127.0.0.1 --port=3034
```

---

这是一个兼容 OpenAI 的 API 代理，旨在：

- **重写请求参数**：调整 API 请求中发送的参数。
- **增加系统提示词**：添加系统提示词以引导模型的行为。
- **增加工具**：比如打开gemini的grounding search
- **根据上下文 Token 长度路由请求**：类似于 moonshot-v1-auto，它会根据输入的 Token 数量将请求路由到不同的模型端点（例如，moonshot-v1-8k、moonshot-v1-32k、moonshot-v1-128k）。

此代理适合部署在 one-api、one-hub 或 new-api 等中转平台的前端。

您可以在 `config.json` 文件中自定义各种 Token 长度和模型的规则。

```bash
npm install
# 编辑 `config.json` 以匹配您的特定需求。
vim config.json
node rewrite.js --host-127.0.0.1 --port=3034
```
