from dotenv import load_dotenv
load_dotenv()

import os

from openai import OpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_MODEL_VISION = os.getenv('OPENAI_MODEL_VISION')
OPENAI_MODEL_TEXT = os.getenv('OPENAI_MODEL_TEXT')
OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS'))

TOKEN_BUFFER = 500

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def invoke_with_base64_image(prompt: str, mime_type: str, base64_image: str) -> str:
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/{mime_type};base64,{base64_image}'}}
            ]
        }
    ]
    return _invoke(messages, OPENAI_MODEL_VISION, OPENAI_MAX_TOKENS - TOKEN_BUFFER)

def invoke_with_text(prompt: str, text: str, temperature: int=0.5) -> str:
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': f'{prompt} {text}'}
            ]
        }
    ]
    return _invoke(messages, OPENAI_MODEL_TEXT, OPENAI_MAX_TOKENS, temperature)

def _invoke(messages: list[dict], model: str, max_tokens: int, temperature: int = 0) -> str:
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        temperature=temperature
    )
    if not response.choices:
        error = response.model_extra.get('error') if response.model_extra else None
        if error:
            raise Exception(f'Error calling {model}: {error}')
        else:
            raise Exception(f'Unknown Error calling {model}')
    answer = response.choices[0].message.content
    print(answer)
    return answer
