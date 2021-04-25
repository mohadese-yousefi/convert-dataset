# convert-dataset
Convert dataset for famous model.

#### How to use
```bash
python convert_yolo_to_efficientdet.py --yolo_dataset_dir </path/to/dataset/> --list_class_names 'class_name0, class_name1' --output_file label.csv
```

Convert yolo dataset to VOC dataset format
```
python convert_yolo_to_VOCformat.py --base_dir </path/to/dataset>
```
`base_dir`: directory of yolo dataset contain image and label (text) file. 
