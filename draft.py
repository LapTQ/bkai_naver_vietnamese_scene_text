import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from utils.plots import *
from utils.general import *

import os

os.system('cat dataset/train_data/training_gt/gt_img_2.txt')

img_path = 'dataset/train_data/training_img/img_2.jpg'
gt_path = img_path.replace('training_img/img_', 'training_gt/gt_img_').replace('.jpg', '.txt')

# image = np.array(Image.open(img_path).convert('RGB'))
# with open(gt_path, 'r') as f:
#     annotations = f.read().split('\n')[:-1]
#     annotations = [line.split(',') for line in annotations]
#     annotations = [[(int(line[_]) if _ < len(line) - 1 else line[_]) for _ in range(len(line))] for line in annotations]
#
#     boxes = [line[:-1] for line in annotations]
#     boxes = np.array(boxes)
#     print(boxes)
# plot_image(image, boxes=boxes)
# plt.show()

# buffer = []
# for gt_path in Path('dataset/train_data/training_gt').glob('*'):
#     with open(gt_path, 'r') as f:
#         annotations = f.read().split('\n')[:-1]
#         annotations = [line.split(',') for line in annotations]
#         annotations = [[(int(line[_]) if _ < 8 else line[_]) for _ in range(len(line))] for line in annotations]
#
#         show = False
#         for line in annotations: #boxes = [line[:-1] for line in annotations]
#             # print(line[:8])
#             # buffer.append(len(set(line[:8])))
#             if len(line) != 9:
#                 show = True
#         if show:
#             print(gt_path)
#             os.system('cat ' + str(gt_path))
#             print(general.process_txt(gt_path))
#
#             # plt.imshow(plt.imread(str(gt_path).replace('training_gt/gt_img_', 'training_img/img_').replace('.txt', '.jpg')))
#             # plt.show()



vocab = []
for gt_path in Path('dataset/train_data/training_gt').glob('*'):
    boxes = process_txt(gt_path)

    # # check process_txt
    # r_input = '\n'.join([','.join(str(_) for _ in line) for line in boxes])
    # try:
    #     assert r_input == open(gt_path, 'r').read(), 'Wrong!!!'
    # except Exception as e:
    #     print(f'{type(e).__name__}: {e}')
    #     os.system('cat ' + str(gt_path))
    #     print(r_input)
    #     exit()

    for box in boxes:
        vocab.append(box[-1])
index_to_word_mapping = dict(enumerate(vocab))
word_to_index_mapping = {word: id for id, word in index_to_word_mapping.items()}
print(word_to_index_mapping)

# gt_path = np.random.choice(list(Path('dataset/train_data/training_gt').glob('*')))
gt_path = 'dataset/train_data/training_gt/gt_img_1.txt'
img_path = str(gt_path).replace('training_gt/gt_img_', 'training_img/img_').replace('.txt', '.jpg')
image = np.array(Image.open(img_path).convert('RGB'))
boxes = process_txt(gt_path)
# print(boxes)
boxes = [[(_ if isinstance(_, int) else word_to_index_mapping[_]) for _ in box] for box in boxes]
boxes = np.array(boxes)
plot_image(image, boxes=boxes, index_to_word_mapping=index_to_word_mapping)
plt.show()

boxes = corner_to_pascal_voc(boxes)
plot_image(image, boxes=boxes, index_to_word_mapping=index_to_word_mapping)
plt.show()
# print(boxes)


