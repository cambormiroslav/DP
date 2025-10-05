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

def get_array_of_test_names_and_path():
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
    results = model.train(data=dataset_yaml, epochs=count_of_epochs, imgsz=32, project = model_train_dir)

def test_yolo():
    pass

def load_and_measure(dir_path, first_ticket, latest_file):
    i = first_ticket - 1
    array_of_images = os.listdir(dir_path)
    while(True):
        file = array_of_images[i]
        start_datetime = datetime.datetime.now()
        #ocr read
        end_datetime = datetime.datetime.now()

        #check data

        diff_datetime = end_datetime - start_datetime
        diff_datetime_seconds = diff_datetime.total_seconds()

        #save to file and print 

        i += 1

        print("Receipt: ", i)

        if i == latest_file:
            break

if __name__ == "__main__":
    train_yolo("yolo12n.pt", dataset_yaml, 600,"./output/yolo12n/")