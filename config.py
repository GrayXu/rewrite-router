# example config

# FOR rewrite_proxy.py
BACKEND_URL = "http://localhost:3000"

# configure routing strategies for different models
ROUTING_RULES = {
    "gpt-4o-auto": {
        'models': {
            8192: "gpt-4o-8k",
            131072: "gpt-4o-2024-08-06"
        },
        'threshold': 0.65
    },
    "gpt-4o-mini-auto": {
        'models': {
            8192: "gpt-4o-mini-8k",
            131072: "gpt-4o-mini"
        },
        'threshold': 0.65
    },
    "Qwen/Qwen2.5-72B-Instruct-auto": {
        'models': {
            32768: "Qwen/Qwen2.5-72B-Instruct",
            131072: "Qwen/Qwen2.5-72B-Instruct-128K"
        },
        'threshold': 0.7
    }
}


REWRITE_RULES = {
    # rewrite request parameters
    'yi-lightning': {
        "max_tokens": 5000
    },
    
    'o1-mini': {
        "stream": "false",
    },
    
    # extend thinking!
    'claude-3-7-sonnet-20250219-thinking': {
        "model": "claude-3-7-sonnet-20250219",
        "thinking": {
            "type": 'enabled',
            "budget_tokens": 8192,
        }
    },
    
    # insert tools (e.g. force google search)
    'gemini-2.0-pro-search': {
        "model": "gemini-2.0-pro",  # change model name
        "message": {
            "tools": [
                {"googleSearch": {}}
            ]
        }
    },
    'gemini-2.0-flash-search': {
        "model": "gemini-2.0-flash",
        "message": {
            "tools": [
                {"googleSearch": {}}
            ]
        }
    },
    'gemini-2.0-flash-thinking-search': {
        "model": "gemini-2.0-flash-thinking",
        "message": {
            "tools": [
                {"googleSearch": {}}
            ]
        }
    },
    # insert system prompts
    'model-with-sys-prompts': {
        "message": [
            {
                "role": "system",
                "content": "You are ChatGPT."
            }
        ]
    },
}