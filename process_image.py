import base64
import os
import requests
import tempfile

from io import BytesIO
from PIL import Image, ImageFile

EXTENSION_TO_PIL_FORMAT = {
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.png': 'PNG',
    '.bmp': 'BMP',
    '.gif': 'GIF',
    '.tiff': 'TIFF',
    '.webp': 'WEBP'
}

def image_to_base64(image_path: str) -> tuple[str, str]:
    with Image.open(image_path) as img:
        format = img.format or EXTENSION_TO_PIL_FORMAT.get(_get_file_ext(image_path), 'PNG')
        base64_image = _get_base64_image(img, format)
        return format.lower(), base64_image

def url_to_base64(url: str, format='PNG') -> str:
    response = requests.get(url, stream=True)
    response.raise_for_status() # raise HTTPError for bad responses (4xx or 5xx)
    with Image.open(BytesIO(response.content)) as img:
        base64_image = _get_base64_image(img, format)
        base64_image_data_url = f'data:image/{format};base64,{base64_image}'
        return base64_image_data_url

def _get_base64_image(img: ImageFile, format: str) -> str:
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.read()).decode()
    return base64_image

def _get_file_ext(image_path: str) -> str:
    return os.path.splitext(image_path)[-1].lower()

def url_to_temp_file(url: str, prefix: str, chunk_size=32*1024, suffix='.png') -> str:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False) as temp_file:
        temp_filepath = temp_file.name
        for chunk in response.iter_content(chunk_size=chunk_size):
            temp_file.write(chunk)
    return temp_filepath
