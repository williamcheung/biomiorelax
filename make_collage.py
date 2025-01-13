from PIL import Image

def create_collage(image_paths: list[str], output_path: str) -> None:
    images = [Image.open(path) for path in image_paths if path]
    num_images = len(images)

    first_image_width, first_image_height = images[0].size
    aspect_ratio = first_image_width / first_image_height

    cols = 2
    if num_images == 1:
        collage_width, collage_height = first_image_width, first_image_height
    else:
        collage_width = first_image_width * cols
        collage_height = int(collage_width / aspect_ratio)

    collage = Image.new('RGB', (collage_width, collage_height))

    x_offset = 0
    y_offset = 0
    for i, img in enumerate(images):
        if i == 0:
            collage.paste(img, (x_offset, y_offset))
            x_offset += img.width
        else:
            target_width = first_image_width
            target_height = int(target_width / aspect_ratio)
            img = img.resize((target_width, target_height))
            collage.paste(img, (x_offset, y_offset))
            x_offset += img.width

        if (i + 1) % cols == 0:
            x_offset = 0
            y_offset += target_height

    collage.save(output_path)
