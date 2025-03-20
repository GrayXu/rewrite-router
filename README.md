# Rewrite router

This is an OpenAI-compatible API router designed to:

- **Rewrite request parameters**: Adjust parameters sent in API requests.
- **Add system prompts**: Add system prompts to guide the model's behavior.
- **Add tools**: Add tools, such as enabling Gemini's grounding search.
- **Route requests based on context token length**: Similar to moonshot-v1-auto, it intelligently routes requests to different model endpoints (e.g., `moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k`) depending on the input's token count.
This proxy is suited for deployment in front of intermediary platforms like one-api, one-hub, or new-api.

You can customize rules for various token lengths and models in the `config.py` file.

```bash
pip install -r requirements.txt

# Edit `config.py` to match your specific needs.
vim config.py

# Run
python ./rewrite.py --port=3034
# Alternatively, use Gunicorn for production deployments (pip install gunicorn):
gunicorn --workers 2 --threads 2 --bind 0.0.0.0:3034 rewrite:app
```

nodejs version
```bash
node rewrite.js
```

---

这是一个兼容 OpenAI 的 API 代理，旨在：

- **重写请求参数**：调整 API 请求中发送的参数。
- **增加系统提示词**：添加系统提示词以引导模型的行为。
- **增加工具**：比如打开gemini的grounding search
- **根据上下文 Token 长度路由请求**：类似于 moonshot-v1-auto，它会根据输入的 Token 数量将请求路由到不同的模型端点（例如，moonshot-v1-8k、moonshot-v1-32k、moonshot-v1-128k）。

此代理适合部署在 one-api、one-hub 或 new-api 等中转平台的前端。

您可以在 `config.py` 文件中自定义各种 Token 长度和模型的规则。

```bash
pip install -r requirements.txt

# 编辑 `config.py` 以匹配您的特定需求。
vim config.py

# 运行
python ./rewrite.py --port=3034
# 或者，使用 Gunicorn 进行生产部署 (pip install gunicorn)：
gunicorn --workers 2 --threads 2 --bind 0.0.0.0:3034 token_router:app
```

nodejs版本
```bash
node rewrite.js