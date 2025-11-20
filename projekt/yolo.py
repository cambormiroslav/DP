import datetime
from ultralytics import YOLO
import os
import cv2
import psutil
import pynvml
import threading

import functions

correct_data_path = "../data_for_control/dataset_objects_correct_data.json"
type_of_data = "objects"
test_images_dir_path = "../dataset/yolo_dataset/test/images/"
dataset_yaml = "../dataset/yolo_dataset/data.yaml"

def test_img(img_path, model, model_name, file_name):
    #get process id
    pid = os.getpid()
    process = psutil.Process(pid)

    #GPU init
    gpu_handle = None
    base_vram_mb = 0.0
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        base_vram_mb = info.used / (1024 * 1024)
        gpu_is_available = True
    except pynvml.NVMLError:
        print("NVIDIA GPU not found.")
        gpu_is_available = False

    #init of thread
    functions.monitor_data["is_running"] = True
    monitor_thread = threading.Thread(
        target=functions.monitor_memory_gpu_vram, 
        args=(process, gpu_handle),
        daemon=True #stops if main script stops
    )
    monitor_thread.start()
    vram_after = 0.0

    #cpu and memory before test model
    process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)

    try:
        image = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        results = model.predict(img_rgb)
        firstResult = results[0]

        boxes = firstResult.boxes
        classes = boxes.cls.cpu().numpy().astype('uint')
        class_names_array = []
        for j in range(len(classes)):
            classId = classes[j]
            className = firstResult.names[classId]
            class_names_array += [className]
    finally:
        # stop thread
        functions.monitor_data["is_running"] = False
        monitor_thread.join(timeout=1.0)
        if gpu_is_available:
            vram_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            vram_after = vram_info.used / (1024 * 1024)
            pynvml.nvmlShutdown() #shutdown nvml

    #get cpu and ram usage
    mem_after = process.memory_info().rss / (1024 * 1024)
    peak_ram_mb = functions.monitor_data["peak_rss_mb"]
    cpu_usage = process.cpu_percent(interval=None)
    
    peak_ram_mb = max(peak_ram_mb, mem_after) #maximum of peak RAM and final value of RAM
    ram_usage = peak_ram_mb - mem_before

    #GPU VRAM usage
    if gpu_is_available:
        total_vram_mb = max(functions.monitor_data["peak_vram_mb"], vram_after) - base_vram_mb
    else:
        total_vram_mb = -1

    functions.save_to_file_cpu_gpu(model_name, type_of_data, True, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                       ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb,
                                       0) #this information is in other file there

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
    
    #GPU init
    gpu_handle = None
    base_vram_mb = 0.0
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        base_vram_mb = info.used / (1024 * 1024)
        gpu_is_available = True
    except pynvml.NVMLError:
        print("NVIDIA GPU not found.")
        gpu_is_available = False

    functions.monitor_data["is_running"] = True
    monitor_thread = threading.Thread(
        target=functions.monitor_memory_gpu_vram, 
        args=(process, gpu_handle),
        daemon=True #stops if main script stops
    )
    monitor_thread.start()
    vram_after = 0.0

    #cpu and memory before test model
    process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)
    start_datetime = datetime.datetime.now()

    try:
        #train model
        model.train(data=dataset_yaml, epochs=count_of_epochs, imgsz=32, project = model_train_dir)
    finally:
            # stop thread
            functions.monitor_data["is_running"] = False
            monitor_thread.join(timeout=1.0)
            if gpu_is_available:
                vram_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                vram_after = vram_info.used / (1024 * 1024)
                pynvml.nvmlShutdown() #shutdown nvml

    end_datetime = datetime.datetime.now()
    #get cpu and ram usage
    mem_after = process.memory_info().rss / (1024 * 1024)
    peak_ram_mb = functions.monitor_data["peak_rss_mb"]
    cpu_usage = process.cpu_percent(interval=None)

    peak_ram_mb = max(peak_ram_mb, mem_after) #maximum of peak RAM and final value of RAM
    ram_usage = peak_ram_mb - mem_before

    #GPU VRAM usage
    if gpu_is_available:
        total_vram_mb = max(functions.monitor_data["peak_vram_mb"], vram_after) - base_vram_mb
    else:
        total_vram_mb = -1

    #time of train
    diff_datetime = end_datetime - start_datetime
    diff_datetime_seconds = diff_datetime.total_seconds()
    
    functions.save_to_file_cpu_gpu(model_specification.replace(".pt", ""), type_of_data, True, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                       ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb,
                                       diff_datetime_seconds)

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
    model = train_yolo("yolo11n.pt", dataset_yaml, 600,"./output_objects/yolo11n/")
    """load_and_measure(model, "yolo11n")

    model = train_yolo("yolo11s.pt", dataset_yaml, 600,"./output_objects/yolo11s/")
    load_and_measure(model, "yolo11s")

    model = train_yolo("yolo11m.pt", dataset_yaml, 600,"./output_objects/yolo11m/")
    load_and_measure(model, "yolo11m")

    model = train_yolo("yolo11l.pt", dataset_yaml, 600,"./output_objects/yolo11l/")
    load_and_measure(model, "yolo11l")

    model = train_yolo("yolo11x.pt", dataset_yaml, 600,"./output_objects/yolo11x/")
    load_and_measure(model, "yolo11x")"""