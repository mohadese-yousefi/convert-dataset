import os
import sys
import shutil
import argparse
import json

import numpy as np
import cv2


def yolo_to_json(args):
    labels = os.listdir(args.yolo_dataset_dir)
    labels = [x for x in labels if x.split('.')[-1] == 'txt']
    for lbl in labels:
        name = lbl.split('.txt')[0]
        print(f'Convering {name}')
        image = cv2.imread(f'{args.yolo_dataset_dir}/{name}.jpg')

        (H, W) = image.shape[:2]

        with open(f'{args.yolo_dataset_dir}/{lbl}', 'r') as l:
            line = l.readline()
            new_label = dict(version="4.4.0", flags= {}, \
                             shapes=list(), \
                               imagePath= f'{name}.jpg',imageData= None, imageHeight= H, \
                             imageWidth= W)
            while line:
                points = line.replace(' ', ',').strip().split(',')
                points = [float(i) for i in points]
                box = np.array(points[1:5]) * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                xmin = int(centerX - (width / 2))
                ymin = int(centerY - (height / 2))
                xmax = int(centerX + (width / 2))
                ymax = int(centerY + (height / 2))

                new_label['shapes'].append({"label": "1", 'points': \
                                            [[xmin, ymin], [xmax, ymin], \
                                            [xmax, ymax], [xmin, ymax]], \
                                            "group_id": None, "shape_type": "polygon","flags": {}})

                line = l.readline()

            with open(f'{args.yolo_dataset_dir}/{name}.json', 'w+') as csv:
                json.dump(new_label, csv, indent=2)



def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_dataset_dir', help='Absolute path of directory include images and label file. ', type=str, required=True)

    return parser.parse_args(argv)


if __name__ == '__main__':
    yolo_to_json(parse_arguments(sys.argv[1:]))

