from flask import Flask, request, Response, stream_with_context
import requests
import json

app = Flask(__name__)

# config
ENDPOINT = "https://api.lingyiwanwu.com/v1/chat/completions"
REWRITE_DATA = {
    "max_tokens": 5000,
}

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Handle chat completion requests"""
    request_data = request.get_json()

    if not request_data or 'messages' not in request_data or not isinstance(request_data['messages'], list):
        return Response("Invalid request: 'messages' field is missing or not a list", status=400)

    # Extract prompt
    prompt = ""
    for message in request_data['messages']:
        prompt += message.get("content", "")

    # rewrite request_data
    for key, value in REWRITE_DATA.items():
        request_data[key] = value

    # Forward request to the target URL
    target_url = ENDPOINT
    headers = {
        'Authorization': request.headers.get('Authorization'),
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(target_url, json=request_data, headers=headers, stream=True)
        response.raise_for_status()

        def generate():
            for line in response.iter_lines():
                if line:
                    yield f"{line.decode('utf-8')}\n\n"

        return Response(stream_with_context(generate()), content_type='text/event-stream')

    except requests.exceptions.RequestException as e:
        error_message = f"Error forwarding request: {e}"
        print(error_message)
        return Response(error_message, status=502)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)