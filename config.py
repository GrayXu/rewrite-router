# example config 
# configure routing strategies for different models
MODEL_ROUTING = {
    "gpt-4o": {
        "endpoints": {
            8192: {"url": "http://localhost:3000/v1/chat/completions", "model": "gpt-4o-8k"},
            131072: {"url": "http://localhost:3000/v1/chat/completions", "model": "gpt-4o"}
        },
        "threshold": 0.9
    },
    "gpt-4o-mini": {
        "endpoints": {
            8192: {"url": "http://localhost:3000/v1/chat/completions", "model": "gpt-4o-mini-8k"},
            131072: {"url": "http://localhost:3000/v1/chat/completions", "model": "gpt-4o-mini"}
        },
        "threshold": 0.9
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "endpoints": {
            32768: {"url": "http://localhost:3000/v1/chat/completions", "model": "Qwen/Qwen2.5-72B-Instruct"},
            131072: {"url": "http://localhost:3000/v1/chat/completions", "model": "Qwen/Qwen2.5-72B-Instruct-128K"}
        },
        "threshold": 0.9
    },
}

# Default routing strategy (used when the requested model is not configured)
DEFAULT_ROUTING = {
    "url": "http://localhost:3000/v1/chat/completions",  # You can set a default forwarding address
    "threshold": 0.9
}