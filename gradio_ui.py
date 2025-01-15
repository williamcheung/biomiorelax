from dotenv import load_dotenv
load_dotenv()

import asyncio
import gradio as gr
import os
import tempfile
import time

from concurrent.futures import ThreadPoolExecutor

from gen_image import ContentPolicyViolation, generate_image
from llm import invoke_with_base64_image, invoke_with_text
from make_collage import create_collage
from process_image import image_to_base64, url_to_temp_file
from utils import UTF8_ENCODING, load_prompt, tuple_if_more_than_one_element

TITLE = 'Bio Mio'

GENERATE_IMAGE_PROMPTS = ['generate_image.prompt.txt']
COLLAGES_DIR = 'collages'

CONTENT_POLICY_VIOLATION_IMAGE_FILE = 'images/content_policy_violation.png'

SAMPLES_DIR = './samples'
sample_images = [os.path.join(SAMPLES_DIR, filename) for filename in os.listdir(SAMPLES_DIR) if filename.endswith(('jpg', 'jpeg', 'png'))]
sample_choices = [('', None)] + [(f'sample {i}', sample_image) for i, sample_image in enumerate(sample_images, start=1)]

async def process_image(image_path: str):
    if not image_path:
        yield tuple_if_more_than_one_element([None] * len(GENERATE_IMAGE_PROMPTS))
        return

    mime_type, base64_image = image_to_base64(image_path)

    img_desc = invoke_with_base64_image(load_prompt('describe_landscape.prompt.txt'), mime_type, base64_image)
    if 'not a landscape' in img_desc.casefold():
        yield tuple_if_more_than_one_element([None] * len(GENERATE_IMAGE_PROMPTS))
        return

    async def generate_images_in_parallel():
        images = [None] * len(GENERATE_IMAGE_PROMPTS)

        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor,
                    lambda i, prompt: generate_image_and_create_temp_file(i, prompt),
                    i, prompt
                )
                for i, prompt in enumerate(GENERATE_IMAGE_PROMPTS, start=1)
            ]

        completed = 0
        for completed_task in asyncio.as_completed(tasks):
            i, image = await completed_task
            images[i] = image
            yield tuple_if_more_than_one_element(images)
            completed += 1
            if completed < len(GENERATE_IMAGE_PROMPTS):
                await asyncio.sleep(1)

    def generate_image_and_create_temp_file(i: int, prompt: str) -> tuple[int, str]:
        image_prompt = load_prompt(prompt)
        image_prompt += img_desc
        try:
            image_url = generate_image(image_prompt.strip())
            image_file = url_to_temp_file(image_url, 'biomio_')

            img_summary = generate_summary(img_desc)
            open(f'{image_file}.txt', 'w', encoding=UTF8_ENCODING).write(img_summary)

        except ContentPolicyViolation as e:
            print(f'ContentPolicyViolation for {e.prompt=} {e}')
            image_file = CONTENT_POLICY_VIOLATION_IMAGE_FILE

        return i-1, image_file

    def generate_summary(img_desc: str) -> str:
        try:
            answer = invoke_with_text(load_prompt('summarize_landscape.prompt.txt'), img_desc)
        except:
            answer = ''
        return answer

    async for images in generate_images_in_parallel():
        yield images

def get_image_summary(image_file: str) -> str:
    if not image_file:
        return ''

    summary_file = f'{os.path.join(tempfile.gettempdir(), os.path.basename(image_file))}.txt'
    try:
        image_summary = open(summary_file, 'r', encoding=UTF8_ENCODING).read()
        return image_summary
    except Exception as e:
        print(f'Error reading {summary_file}: {e}')
        return ''

def set_image_summary_to_read(mute: bool, summary: str) -> str:
    return None if mute else summary

def make_collage(image0: str, image1: str) -> dict:
    if not image0:
        return gr.update(visible=False)

    image_paths = [image0, image1]
    image_paths += [image1, image0]
    collage_path = f'{COLLAGES_DIR}/biomio_collage_{time.time_ns()}.jpg'
    create_collage(image_paths, collage_path)
    return gr.update(value=collage_path, visible=True)

def update_image_from_sample(selected_sample: str) -> str:
    return selected_sample

with gr.Blocks(title=TITLE, theme=gr.themes.Monochrome(), css='''
    footer {visibility: hidden}

    /* make container full width */
    .gradio-container {
        width: 100% !important; /* flll width */
        max-width: 100% !important; /* prevent max-width restriction */
        margin: 5px 0px 5px 0px !important; /* top, right, bottom, left */
    }

    /* inner row vertical alignment */
    #pin-input-centered-row > div > div {
        display: flex;
        align-items: center;
    }
            ''') as demo:

    # JavaScript for text-to-speech
    tts_js = '''
    async (text) => {
        console.log('text: ' + text);
        if (!text) {
            return;
        }

        const voices = await new Promise((resolve) => {
            if (window.speechSynthesis.getVoices().length !== 0) {
                resolve(window.speechSynthesis.getVoices());
            } else {
                window.speechSynthesis.onvoiceschanged = () => {
                    resolve(window.speechSynthesis.getVoices());
                };
            }
        });

        const femaleVoice = voices.find(voice =>
            voice.name.toLowerCase().includes("female") ||
            voice.name.toLowerCase().includes("woman")
        );
        if (!femaleVoice) {
            console.log('No recognized female voice found; using default voice.');
        }

        let chunks = [text];

        const userAgent = navigator.userAgent;
        const vendor = navigator.vendor;
        console.log('userAgent: ' + userAgent);
        console.log('vendor: ' + vendor);
        if (/Google/.test(vendor) && /Chrome/.test(userAgent) && !/Edg/.test(userAgent)) {
            chunks = text.split('.').map(sentence => sentence.trim()).filter(sentence => sentence.length > 0);
        }

        for (let i = 0; i < chunks.length; i++) {
            await new Promise((resolve, reject) => {
                console.log('chunk: ' + chunks[i]);
                const utterance = new SpeechSynthesisUtterance(chunks[i]);
                if (femaleVoice) {
                    utterance.voice = femaleVoice;
                }
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utterance);

                utterance.onend = () => {
                    resolve();
                };
                utterance.onerror = (error) => {
                    resolve();
                };
            });
        }
    }
    '''

    with gr.Row():
        gr.Markdown(f'# {TITLE} ... relax')
        collage_button = gr.Button(f'make collage', scale=0)

    with gr.Row():
        collage_file = gr.File(label='Collage', visible=False)

    with gr.Row():
        sample_selector = gr.Dropdown(label='Select a Sample', choices=sample_choices, interactive=True)

    with gr.Row():
        upload_image = gr.Image(label='Landscape', type='filepath')
        output_image = gr.Image(label='Biomio', type='filepath', interactive=False)

    with gr.Row():
        with gr.Column():
            image_summary = gr.Textbox(label='Imagine', show_copy_button=True, interactive=False, autoscroll=False, lines=2)
            image_summary_to_read = gr.Textbox(visible=False)
            mute_checkbox = gr.Checkbox(label='🔇', info='mute', value=False)


    gr.HTML(
        '''
        <div id='app-footer' style='text-align: center;'>
            Powered by <a href='https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct' target='_blank'>Qwen2 Vision</a> and <a href='https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407' target='_blank'>Mistral Nemo</a>
        </div>
        <div id='app-footer' style='text-align: center;'>
            on <a href='https://studio.nebius.ai' target='_blank'>Nebius AI Studio</a> 🚀
        </div>
        <br/>
        <div id='app-footer' style='text-align: center;'>
            With sample images from <a href='https://www.kaggle.com/datasets/arnaud58/landscape-pictures' target='_blank'>Kaggle Datasets</a>
        </div>
        '''
    )

    upload_image.change(
        fn=lambda: gr.update(visible=False),
        inputs=None,
        outputs=collage_file
    ).then(
        fn=lambda: gr.update(interactive=False),
        inputs=None,
        outputs=sample_selector
    ).then(
        fn=process_image,
        inputs=upload_image,
        outputs=[output_image]
    ).then(
        fn=lambda: gr.update(interactive=True),
        inputs=None,
        outputs=sample_selector
    ).then(
        fn=get_image_summary,
        inputs=output_image,
        outputs=image_summary
    ).then(
        fn=set_image_summary_to_read,
        inputs=[mute_checkbox, image_summary],
        outputs=image_summary_to_read
    ).then(
        None,
        inputs=image_summary_to_read,
        outputs=None,
        js=tts_js
    )

    sample_selector.change(fn=update_image_from_sample, inputs=sample_selector, outputs=upload_image)

    collage_button.click(fn=make_collage, inputs=[upload_image, output_image], outputs=collage_file)

demo.launch(server_name='0.0.0.0')
