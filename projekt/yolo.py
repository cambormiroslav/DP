import datetime
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import os
import random
import matplotlib.pyplot as plt
import cv2

correct_data_path = "../data_for_control/dataset_objects_correct_data.json"
type_of_data = "objects"
test_images_dir_path = "../dataset/yolo_dataset/test/"
labels_dir_path = "../dataset/yolo_dataset/labels/"
dataset_yaml = "../dataset/yolo_dataset/data.yaml"

def test_img(img_path, model, file_name):
    image = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = model.predict(img_rgb)
    firstResult = results[0]

    boxes = firstResult.boxes
    classes = boxes.cls.numpy().astype('uint')
    class_names_array = []
    for j in range(len(classes)):
        classId = classes[j]
        className = firstResult.names[classId]
        class_names_array += [className]
    
    return {file_name: class_names_array}

def get_array_of_test_names_and_paths():
    array_of_file_paths = []
    array_file_names = []
    for file_path in os.listdir(test_images_dir_path):
        array_of_file_paths += [test_images_dir_path + file_path]
        array_file_names += [file_path]
    
    return (array_file_names, array_of_file_paths)

def train_yolo(model_specification, dataset_yaml, count_of_epochs, model_train_dir):
    model = YOLO(model_specification)
    if not os.path.exists(model_train_dir):
        os.mkdir(model_train_dir)
    model.train(data=dataset_yaml, epochs=count_of_epochs, imgsz=32, project = model_train_dir)

    return model

def load_and_measure(model):
    arrays_of_test_files = get_array_of_test_names_and_paths()
    for index in range(len(arrays_of_test_files[0])):
        start_datetime = datetime.datetime.now()
        print(test_img(arrays_of_test_files[1][index], model, arrays_of_test_files[0][index]))
        end_datetime = datetime.datetime.now()

        diff_datetime = end_datetime - start_datetime
        diff_datetime_seconds = diff_datetime.total_seconds()

        print("Receipt: ", arrays_of_test_files[0][index])

if __name__ == "__main__":
    model = train_yolo("yolo12n.pt", dataset_yaml, 50,"./output/yolo12n/") #600
    load_and_measure(model)