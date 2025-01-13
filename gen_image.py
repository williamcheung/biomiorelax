from dotenv import load_dotenv
load_dotenv()

import os

from openai import BadRequestError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_never, wait_exponential

DALL_E_OPENAI_API_KEY = os.getenv('DALL_E_OPENAI_API_KEY')
DALL_E_OPENAI_BASE_URL = os.getenv('DALL_E_OPENAI_BASE_URL')

DALL_E_MODEL = os.getenv('DALL_E_MODEL')
DALL_E_MAX_PROMPT_LEN = int(os.getenv('DALL_E_MAX_PROMPT_LEN'))

IMAGE_GEN_RATE_LIMIT_PER_MIN = int(os.getenv('IMAGE_GEN_RATE_LIMIT_PER_MIN'))
IMAGE_GEN_MIN_WAIT_SECS = int(60 / IMAGE_GEN_RATE_LIMIT_PER_MIN) + 1

images_client = OpenAI(api_key=DALL_E_OPENAI_API_KEY, base_url=DALL_E_OPENAI_BASE_URL).images

class ContentPolicyViolation(Exception):
    def __init__(self, message: str, prompt: str):
        super().__init__(message)
        self.prompt = prompt

@retry(wait=wait_exponential(min=IMAGE_GEN_MIN_WAIT_SECS), stop=stop_never, retry=retry_if_exception_type(RateLimitError))
def generate_image(image_gen_prompt: str) -> str:
    model = DALL_E_MODEL
    max_prompt_len = DALL_E_MAX_PROMPT_LEN
    print(f'Using model: {model}, max_prompt_len: {max_prompt_len}')

    prompt = image_gen_prompt[:max_prompt_len].strip()
    if len(image_gen_prompt) > max_prompt_len:
        print(f'Using truncated prompt: {prompt}')

    try:
        response = images_client.generate(model=model, prompt=prompt)
        image_url = response.data[0].url
        print(image_url)
        return image_url
    except RateLimitError as e:
        print(f'Error generating image: {e}')
        print('Will wait and retry due to rate limit error...')
        raise e
    except BadRequestError as e:
        if e.code == 'content_policy_violation':
            raise ContentPolicyViolation(e.message, prompt)
        else:
            raise
