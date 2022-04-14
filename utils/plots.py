import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def plot_image(image, **kwargs):
    """

    :param image: numpy array of shape (H, W, C) representing image in RGB color space.
    :param kwargs:
        boxes: numpy array - like of shape (N, 9) or (N, 10).
        font_size: int
        color: string - boxes color
        index_to_word_mapping: Python dictionary
        border_width: int
    :return:
    """
    assert len(image.shape) == 3, f"Invalid shape for image, must be a 3-dim array, got {image.shape}"
    image = Image.fromarray(image)
    plotted_image = ImageDraw.Draw(image)

    if 'font_size' not in kwargs:
        kwargs['font_size'] = 14
    font = ImageFont.truetype('Arial.ttf', kwargs['font_size'])

    if 'color' not in kwargs:
        kwargs['color'] = 'red'

    if 'border_width' not in kwargs:
        kwargs['border_width'] = 2

    if 'boxes' in kwargs:
        assert kwargs['boxes'].shape[1] in [5, 6, 9, 10], f"Invalid box shape: must be (N, 5) or (N, 6) or (N, 9) or (N, 10), got {kwargs['boxes'].shape}"
        for box in kwargs['boxes']:
            # if boxes is in original format of 4 coordinates
            if kwargs['boxes'].shape[1] in [9, 10]:
                (x1, y1, x2, y2, x3, y3, x4, y4), word_id = box[:8], box[-1]
                plotted_image.line(
                    ((x1, y1), (x2, y2)),
                    fill=kwargs['color'],
                    width=kwargs['border_width']
                )
                plotted_image.line(
                    ((x2, y2), (x3, y3)),
                    fill=kwargs['color'],
                    width=kwargs['border_width']
                )
                plotted_image.line(
                    ((x3, y3), (x4, y4)),
                    fill=kwargs['color'],
                    width=kwargs['border_width']
                )
                plotted_image.line(
                    ((x4, y4), (x1, y1)),
                    fill=kwargs['color'],
                    width=kwargs['border_width']
                )
            # else if boxes is in converted into pascal_voc format
            else:
                (x1, y1, x2, y2), word_id = box[:4], box[-1]
                plotted_image.rectangle(
                    ((x1, y1), (x2, y2)),
                    outline=kwargs['color'],
                    width=kwargs['border_width']
                )

            word = kwargs['index_to_word_mapping'][word_id] if 'index_to_word_mapping' in kwargs else str(word_id)
            msg = ' ' + word + (f': {box[-2]:.2f} ' if kwargs['boxes'].shape[-1] == 10 else ' ')

            text_w, text_h = font.getsize(msg)
            plotted_image.rectangle(((x1, y1 - text_h), (x1 + text_w, y1)), fill=kwargs['color'], outline=kwargs['color'])
            plotted_image.text((x1, y1 - text_h), msg, fill='white', font=font)

    plt.imshow(image)




