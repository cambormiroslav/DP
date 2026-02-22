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
    """
    * Test model
    * Measure the time of run between request and response of model is seconds.
    * Measure CPU/GPU and RAM/VRAM usage

    Input:
        - img_path:
            - path to test image
        - model:
            - model instance
        - model_name:
            - text representation of model
        - file_name:
            - name of test image
    """
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

    detections = []

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

    start_datetime = datetime.datetime.now()

    try:
        image = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        results = model.predict(img_rgb)
        first_result = results[0]

        boxes = first_result.boxes

        #(x_min, y_min, x_max, y_max)
        coords_array = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype('uint')
        confidences = boxes.conf.cpu()
        
        for index, classId in enumerate(classes):
            class_name = first_result.names[classId]
            box_coord = coords_array[index]
            confidence = confidences[index]

            detections.append({
                "class_name": class_name,
                "x_min": box_coord[0],
                "y_min": box_coord[1],
                "x_max": box_coord[2],
                "y_max": box_coord[3],
                "confidence" : confidence
            })
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
    
    #time of test
    diff_datetime = end_datetime - start_datetime
    diff_datetime_seconds = diff_datetime.total_seconds()

    max_iou_detections, good_boxes = functions.get_max_iou_and_good_boxes(file_name, detections)

    for iou_threshold in functions.iou_thresholds:
        map_values = functions.get_mAP(max_iou_detections, good_boxes, iou_threshold)
        functions.save_to_file_object(model_name, type_of_data, map_values["map"],
                                      map_values["map_50"], map_values["map_75"],
                                      map_values["map_large"], map_values["mar_100"],
                                      map_values["mar_large"], iou_threshold)
    functions.save_to_file_object_main(model_name, type_of_data, diff_datetime_seconds, 0)

    functions.save_to_file_cpu_gpu(model_name, type_of_data, True, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                       ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb,
                                       0) #this information is in other file there

    return {file_name: detections}

def get_array_of_test_names_and_paths():
    """
    Getter for test names and paths of images

    Output:
        - array_file_names:
            - file names in array
        - array_of_file_paths
            - file paths in array
    """
    array_of_file_paths = []
    array_file_names = []
    for file_path in os.listdir(test_images_dir_path):
        array_of_file_paths += [test_images_dir_path + file_path]
        array_file_names += [file_path]
    
    return (array_file_names, array_of_file_paths)

def train_yolo(model_specification, dataset_yaml, count_of_epochs, model_train_dir):
    """
    * Test model
    * Measure the time of run between request and response of model is seconds.
    * Measure CPU/GPU and RAM/VRAM usage

    Input:
        - model_specification
            - file model.pt
        - dataset_yaml
            - path to data.yaml file
        - count_of_epochs
        - model_train_dir
            - directory where save data from trainnig model
    """
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
        model.train(data=dataset_yaml, epochs=count_of_epochs, imgsz=890, project = model_train_dir, batch = -1)
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
    
    functions.save_to_file_cpu_gpu(model_specification.replace(".pt", ""), type_of_data, False, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                       ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb,
                                       diff_datetime_seconds)

    return model


def load_and_measure(model, model_name):
    """
    Call testing model

    Input:
        - model:
            - model instance
        - model_name:
            - model text representation
    """
    arrays_of_test_files = get_array_of_test_names_and_paths()
    for index in range(len(arrays_of_test_files[0])):
        test_img(arrays_of_test_files[1][index], model, model_name, arrays_of_test_files[0][index])

if __name__ == "__main__":
    model = train_yolo("yolo11n.pt", dataset_yaml, 600,"./output_objects/yolo11n/")
    load_and_measure(model, "yolo11n")

    model = train_yolo("yolo11s.pt", dataset_yaml, 600,"./output_objects/yolo11s/")
    load_and_measure(model, "yolo11s")

    model = train_yolo("yolo11m.pt", dataset_yaml, 600,"./output_objects/yolo11m/")
    load_and_measure(model, "yolo11m")

    model = train_yolo("yolo11l.pt", dataset_yaml, 600,"./output_objects/yolo11l/")
    load_and_measure(model, "yolo11l")