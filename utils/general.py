import numpy as np

def process_txt(file_path):
    """
    Read .txt file and convert the content to Python nested list.
    :param file_input: Python str containing content of .txt annotation file.
    :return: Python list of [x1 -> int, y1 -> int, x2 -> int, y2 -> int, x3 -> int, y3 -> int, x4 -> int, y4 -> int, label -> str]
    """
    with open(file_path, 'r') as f:
        file_input = f.read()
    f.close()
    lines = file_input.split('\n')[:-1]
    # result = []
    # for line in lines:
    #     buffer = ['']
    #     count = 0
    #     for i in range(len(line)):
    #         if line[i] != ',' or i == len(line) - 1:
    #             buffer[-1] += line[i]
    #         else:
    #             count += 1
    #             if count <= 8:
    #                 buffer[-1] = int(buffer[-1])
    #             buffer.append('')
    #     result.append(buffer)
    lines = [line.split(',') for line in lines]
    result = [[int(_) for _ in line[:8]] + [','.join(line[8:])] for line in lines]
    return result

def corner_to_pascal_voc(boxes):
    """
    Convert boxes from [x1, y1, x2, y2, x3, y3, x4, y4, ...] to [xmin, ymin, xmax, ymax, ...].
    :param boxes: numpy array - like, of shape (N, M) where M >= 8
    :return: numpy array - like, of shape (N, M') where M' = 4 + (M - 8)
    """
    xmins = np.min(
        np.stack([boxes[:, 0], boxes[:, 6]], axis=1),
        axis=1
    )
    ymins = np.min(
        np.stack([boxes[:, 1], boxes[:, 3]], axis=1),
        axis=1
    )
    xmaxs = np.max(
        np.stack([boxes[:, 2], boxes[:, 4]], axis=1),
        axis=1
    )
    ymaxs = np.max(
        np.stack([boxes[:, 5], boxes[:, 7]], axis=1),
        axis=1
    )

    new_boxes = np.concatenate(
        [
            np.stack([xmins, ymins, xmaxs, ymaxs], axis=1),
            boxes[:, 8:]
        ],
        axis=1
    )
    return new_boxes