import datetime
from ultralytics import YOLO
import os
import cv2
import psutil

import functions

correct_data_path = "../data_for_control/dataset_objects_correct_data.json"
type_of_data = "objects"
test_images_dir_path = "../dataset/yolo_dataset/test/"
labels_dir_path = "../dataset/yolo_dataset/labels/"
dataset_yaml = "../dataset/yolo_dataset/data.yaml"

def test_img(img_path, model, model_name, file_name):
    #get process id
    pid = os.getpid()
    process = psutil.Process(pid)
    #memory before test model
    mem_before = process.memory_info().rss / (1024 * 1024)

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
    
    #get cpu and ram usage
    cpu_usage = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)
    ram_usage = mem_after - mem_before

    functions.save_to_file_cpu_gpu(model_name, True, cpu_usage, ram_usage, 0, 0, 0)

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

    #get process id
    pid = os.getpid()
    process = psutil.Process(pid)
    #memory before train model
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    start_datetime = datetime.datetime.now()

    #train model
    model.train(data=dataset_yaml, epochs=count_of_epochs, imgsz=32, project = model_train_dir)

    #get cpu and ram usage
    cpu_usage = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)
    ram_usage = mem_after - mem_before

    end_datetime = datetime.datetime.now()

    #time of train
    diff_datetime = end_datetime - start_datetime
    diff_datetime_seconds = diff_datetime.total_seconds()

    functions.save_to_file_cpu_gpu(model_specification.replace(".pt", ""), False, cpu_usage, ram_usage, 0, 0, diff_datetime_seconds)

    return model

def check_data(data, file_name):
    array_of_types = data[file_name]
    tuple_data_output = (0, 0, 0, 1, {}, [], [])
    for type_object in array_of_types:
        tuple_data = functions.check_the_data_object({"type" : type_object}, file_name, correct_data_path, False)
        if(tuple_data[3] != 1):
            tuple_data_output = tuple_data

    return tuple_data_output


def load_and_measure(model, model_name):
    arrays_of_test_files = get_array_of_test_names_and_paths()
    for index in range(len(arrays_of_test_files[0])):
        start_datetime = datetime.datetime.now()
        response = test_img(arrays_of_test_files[1][index], model, model_name, arrays_of_test_files[0][index])
        end_datetime = datetime.datetime.now()

        diff_datetime = end_datetime - start_datetime
        diff_datetime_seconds = diff_datetime.total_seconds()

        
        data_tuple = check_data(response, arrays_of_test_files[0][index])
        correctness = data_tuple[0]
        correct_data = data_tuple[1]
        incorect_data = data_tuple[2]
        not_found_data = data_tuple[3]
        dict_of_incorect = data_tuple[4]
        array_not_found = data_tuple[5]
        functions.save_to_file_object(model_name, type_of_data, [correctness, correct_data, incorect_data,
                                                                 not_found_data, diff_datetime_seconds],
                                                                 dict_of_incorect, array_not_found)

if __name__ == "__main__":
    model = train_yolo("yolo12n.pt", dataset_yaml, 100,"./output_objects/yolo12n/") #600
    load_and_measure(model, "yolo12n")