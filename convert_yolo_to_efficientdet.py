import os
import sys
import shutil
import argparse

import numpy as np
import cv2


def yolo_to_efficientdet(args):
    labels = os.listdir(args.yolo_dataset_dir)
    labels = [x for x in labels if x.split('.')[-1] == 'txt']
    for lbl in labels:
        name = lbl.split('.')[0]
        image = cv2.imread(f'{args.yolo_dataset_dir}/{name}.jpg')
        (H, W) = image.shape[:2]

        with open(f'{args.yolo_dataset_dir}/{lbl}', 'r') as l:
            line = l.readline()
            while line:
                points = line.replace(' ', ',').strip().split(',')
                points = [float(i) for i in points]
                newline = [f'{args.yolo_dataset_dir}/{name}.jpg']
                box = np.array(points[1:5]) * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                xmin = int(centerX - (width / 2))
                ymin = int(centerY - (height / 2))
                xmax = int(centerX + (width / 2))
                ymax = int(centerY + (height / 2))

                newline.extend([str(xmin), str(ymin), str(xmax), str(ymin), \
                               str(xmax), str(ymax), str(xmin), str(ymax), \
                               args.list_class_names[int(points[0])]])
                final_lable = ','.join(newline)

                with open(f'{args.output_file}', 'a+') as csv:
                    csv.write(f'{final_lable}\n')

                line = l.readline()


def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_dataset_dir', help='Absolute path of directory include images and label file. ', type=str, required=True)
    parser.add_argument('--output_file', help='Save csv file into it. ', type=str, default='label.csv')
    parser.add_argument('--list_class_names', nargs='+', help='Save the label.csv into it. ')

    return parser.parse_args(argv)


if __name__ == '__main__':
    yolo_to_efficientdet(parse_arguments(sys.argv[1:]))

