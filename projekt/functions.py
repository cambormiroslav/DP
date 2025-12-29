import json
import codecs
import psutil
import pynvml
import time
import os
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

pattern_test_dir_output_path = "./output_pattern_test/"
pattern_test_object_dir_output_path = "./output_pattern_test_objects/"
test_dir_path_output = "./output/"
test_dir_objects_path_output = "./output_objects/"

iou_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

monitor_data = {
    "peak_rss_mb": 0.0,
    "peak_cpu_percent": 0.0,
    "peak_vram_mb": 0.0,
    "peak_gpu_utilization": 0,
    "is_running": True
}

def create_dir_if_not_exists(path_to_directory):
    if not os.path.exists(path_to_directory):
        os.makedirs(path_to_directory)

def monitor_memory_gpu_vram(process, gpu_handle):
    monitor_data["peak_rss_mb"] = 0.0
    monitor_data["peak_cpu_percent"] = 0.0
    monitor_data["peak_vram_mb"] = 0.0
    monitor_data["peak_gpu_utilization"] = 0

    # init of CPU percentage usage measurement
    try:
        process.cpu_percent(interval=None) 
    except psutil.NoSuchProcess:
        pass
    
    while monitor_data["is_running"]:
        #CPU and RAM measurement
        total_rss_bytes = 0
        try:
            #CPU percentage usage measurement
            current_cpu = process.cpu_percent(interval=None)
            if current_cpu > monitor_data["peak_cpu_percent"]:
                monitor_data["peak_cpu_percent"] = current_cpu

            #memory of main process
            total_rss_bytes += process.memory_info().rss
            
            #memory of all children recursive
            children = process.children(recursive=True)
            for child in children:
                #children ends quicker
                try:
                    total_rss_bytes += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            
            #to MB
            current_rss_mb = total_rss_bytes / (1024 * 1024)
            
            if current_rss_mb > monitor_data["peak_rss_mb"]:
                monitor_data["peak_rss_mb"] = current_rss_mb
                
        except psutil.NoSuchProcess:
            break

        if gpu_handle:
            try:
                #GPU measurement
                util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                if util.gpu > monitor_data["peak_gpu_utilization"]:
                    monitor_data["peak_gpu_utilization"] = util.gpu
                
                #VRAM measurement
                info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                current_vram_mb = info.used / (1024 * 1024)
                if current_vram_mb > monitor_data["peak_vram_mb"]:
                    monitor_data["peak_vram_mb"] = current_vram_mb
                    
            except pynvml.NVMLError:
                print("NVML error.")
                break
        
        time.sleep(0.01)

def calculate_iou(box_ref, box_test):
    xmin_ref, ymin_ref, xmax_ref, ymax_ref = box_ref
    xmin_test, ymin_test, xmax_test, ymax_test = box_test
    
    max_of_xmin = max(xmin_ref, xmin_test)
    max_of_ymin = max(ymin_ref, ymin_test)
    min_of_xmax = min(xmax_ref, xmax_test)
    min_of_ymax = min(ymax_ref, ymax_test)

    if min_of_xmax < max_of_xmin or min_of_ymax < max_of_ymin:
        return 0
    
    intersection = (min_of_xmax - max_of_xmin) * (min_of_ymax - max_of_ymin)
    size_of_ref = (xmax_ref - xmin_ref) * (ymax_ref - ymin_ref)
    size_of_test = (xmax_test - xmin_test) * (ymax_test - ymin_test)
    union_of_sizes = size_of_ref + size_of_test - intersection

    return intersection / union_of_sizes

def delete_objects_if_something_missing(detections):
    required = ("x_min", "y_min", "x_max", "y_max")
    kept = []

    for detected_object in detections:
        missing = [k for k in required if k not in detected_object]
        if not missing:
            kept.append(detected_object)

    return kept

def get_max_iou_and_good_boxes(file_name, detections):
    good_boxes = get_boxes(file_name)

    correct_format_detections = delete_objects_if_something_missing(detections)

    for detected_object in correct_format_detections:
        box_detected = (detected_object["x_min"], detected_object["y_min"],
                        detected_object["x_max"], detected_object["y_max"])
        max_iou = 0.0
        for good_box in good_boxes:
            iou = calculate_iou(good_box, box_detected)
            if iou > max_iou:
                max_iou = iou
        detected_object["iou"] = max_iou

    return (correct_format_detections, good_boxes)

def get_boxes(file_name):
    correct_data_path = "../data_for_control/dataset_objects_correct_data.json"
    with open(correct_data_path, 'r') as file:
        data = json.load(file)
        boxes = []
        for obj in data[file_name]:
            box = (obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"])
            boxes.append(box)
    return boxes

def load_json_response(response):
    try:
        json_response = json.loads(response)
        if "objects" in json_response:
            return (json_response, 0)
        else:
            return ({"objects": []}, 1)
    except:
        return ({"objects": []}, 2)

def get_predictions_torch(detections):
    boxes = []
    labels = []
    scores = []
    for detection in detections:
        if detection["class_name"] == "person" or detection["name"] == "person":
            labels.append(0) #person
        else:
            labels.append(1) #not person
        boxes.append([detection["x_min"], detection["y_min"], detection["x_max"], detection["y_max"]])
        scores.append(detection["confidence"])

    return[{
        "boxes": torch.tensor(boxes),
        "labels": torch.tensor(labels),
        "scores": torch.tensor(scores)
    }]

def get_target_torch(good_boxes):
    boxes = []
    labels = []
    for good_box in good_boxes:
        labels.append(0) #person
        boxes.append([good_box[0], good_box[1], good_box[2], good_box[3]])

    return[{
        "boxes": torch.tensor(boxes),
        "labels": torch.tensor(labels)
    }]

def get_mAP(iou_detections, good_boxes, iou_threshold):
    mAP_solver = MeanAveragePrecision(box_format='xyxy', iou_type="bbox")

    iou_detections = []
    for detected_object in iou_detections:
        if detected_object["iou"] >= iou_threshold:
            iou_detections.append(detected_object)
    
    predicted_torch = get_predictions_torch(iou_detections)
    target_torch = get_target_torch(good_boxes)

    mAP_solver.update(predicted_torch, target_torch)
    mAP_result = mAP_solver.compute()
    
    return mAP_result

"""
* Check the response characteristics.
* Check corectness of data.
* I not corrected output is: (0, 0, 0, count_of_correct_data, 0, {}, [], []).

Input: (Dictionary model as string, Name of comparing img, Path to correct data file)
Output: (Correctness, Count of correct data, Count of incorrect data, 
        Not founded data in response (main keys), Not founded number of goods,
        Dictionary of incorrect data, Array of not founded data (only keys),
        Array of not founded names goods)
"""
def check_the_data_ocr(dict_model, name_of_file, path_to_correct_data, load_json):
    correct_data_counted = 0
    incorrect_data_counted = 0
    not_in_dict_counted = 0
    goods_not_counted = 0

    dict_incorrect = {}
    array_not_found = []
    array_goods_not = []

    with open(path_to_correct_data, 'r') as file:
        data = json.load(file)[name_of_file]
        
        count_of_data = data["count_of_data"]

        if load_json:
            try:
                dict_model = json.loads(dict_model)
            except:
                return (0, 0, 0, count_of_data, 0, {}, [], [])

        try:
            company_data = dict_model["company"].lower()
            if data["company"].lower() == company_data:
                print("Company Correct")
                correct_data_counted += 1
            else:
                print("Company Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["company"] = company_data
        except:
            if "company" in data:
                print("Company Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["company"]
        
        try:
            address_data = data["address"].lower()
            if data["address"].lower() == address_data:
                print("Address Correct")
                correct_data_counted += 1
            else:
                print("Address Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["address"] = address_data
        except:
            if "address" in data:
                print("Address Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["address"]
        
        try:
            phone_number_data = dict_model["phone_number"]
            if data["phone_number"] == phone_number_data:
                print("Phone Number Correct")
                correct_data_counted += 1
            else:
                print("Phone Number Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["phone_number"] = phone_number_data
        except:
            if "phone_number" in data:
                print("Phone Number Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["phone_number"]
        
        try:
            server_data = dict_model["server"].lower()
            if data["server"].lower() == server_data:
                print("Server Correct")
                correct_data_counted += 1
            else:
                print("Server Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["server"] = server_data
        except:
            if "server" in data:
                print("Server Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["server"]

        try:
            station_data = int(dict_model["station"])
            if data["station"] == station_data:
                print("Station Correct")
                correct_data_counted += 1
            else:
                print("Station Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["station"] = station_data
        except:
            if "station" in data:
                print("Station Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["station"]
        
        try:
            order_number_data = dict_model["order_number"]
            if data["order_number"] == order_number_data:
                print("Order Number Correct")
                correct_data_counted += 1
            else:
                print("Order Number Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["order_number"] = order_number_data
        except:
            if "order_number" in data:
                print("Order Number Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["order_number"]
        
        try:
            table_data = dict_model["table"].lower()
            if data["table"].lower() == table_data:
                print("Table Correct")
                correct_data_counted += 1
            else:
                print("Table Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["table"] = table_data
        except:
            if "table" in data:
                print("Table Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["table"]
        
        try:
            guests_data = int(dict_model["guests"])
            if data["guests"] == guests_data:
                print("Guests Correct")
                correct_data_counted += 1
            else:
                print("Guests Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["guests"] = guests_data
        except:
            if "guests" in data:
                print("Guests Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["guests"]

        try:
            for good in data["goods"]:
                if (good in dict_model["goods"]):
                    correct_data_counted += 1
                    print("Good Correct")
                    try:
                        amount_data = int(dict_model["goods"][good]["amount"])
                        if data["goods"][good]["amount"] == amount_data:
                            print("Amount Correct")
                            correct_data_counted += 1
                        else:
                            print("Amount Incorrect")
                            incorrect_data_counted += 1
                            dict_incorrect["amount"] = amount_data
                    except:
                        if "amount" in data["goods"][good]:
                            print("Amount Not In Dict")
                            not_in_dict_counted += 1
                            array_not_found += ["amount"]
                
                    try:
                        price_data = float(dict_model["goods"][good]["price"])
                        if data["goods"][good]["price"] == price_data:
                            print("Price Correct")
                            correct_data_counted += 1
                        else:
                            print("Price Incorrect")
                            incorrect_data_counted += 1
                            dict_incorrect["price"] = price_data
                    except:
                        if "price" in data["goods"][good]:
                            print("Price Not In Dict")
                            not_in_dict_counted += 1
                            array_not_found += ["price"]
                else:
                    print(f"{good} Incorrect Or Not In File")
                    goods_not_counted += 1
                    array_goods_not += [good]
        except:
            try:
                goods_not_counted = len(data["goods"])
                for good in data["goods"]:
                    array_goods_not += [good]
                    print(f"{good} Incorrect Or Not In File")
            except:
                goods_not_counted = 0
            

        try:
            subtotal_data = float(dict_model["sub_total"])
            if data["sub_total"] == subtotal_data:
                print("Subtotal Correct")
                correct_data_counted += 1
            else:
                print("SubTotal Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["sub_total"] = subtotal_data
        except:
            if "sub_total" in data:
                print("Subtotal Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["sub_total"]
        
        try:
            tax_data = float(dict_model["tax"])
            if data["tax"] == tax_data:
                print("Tax Correct")
                correct_data_counted += 1
            else:
                print("Tax Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["tax"] = tax_data
        except:
            if "tax" in data:
                print("Tax Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["tax"]
        
        try:
            total_data = float(dict_model["total"])
            if data["total"] == total_data:
                print("Total Correct")
                correct_data_counted += 1
            else:
                print("Total Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["total"] = total_data
        except:
            if "total" in data:
                print("Total Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["total"]

        try:
            date_data = dict_model["date"]
            if data["date"] == date_data:
                print("Date Correct")
                correct_data_counted += 1
            else:
                print("Date Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["date"] = date_data
        except:
            if "date" in data:
                print("Date Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["date"]
        
        try:
            time_data = dict_model["time"].lower()
            if time_data == data["time"].lower():
                print("Time Correct")
                correct_data_counted += 1
            else:
                print("Time Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["time"] = time_data
        except:
            if "time" in data:
                print("Time Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["time"]

        if "fax_number" in data:
            try:
                fax_number_data = dict_model["fax_number"].lower()
                if fax_number_data == data["fax_number"].lower():
                    print("Fax Number Correct")
                    correct_data_counted += 1
                else:
                    print("Fax Number Incorrect")
                    incorrect_data_counted += 1
                    dict_incorrect["fax_number"] = time_data
            except:
                print("Fax Number Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["fax_number"]

        correctness = correct_data_counted / count_of_data

        return (correctness, correct_data_counted, incorrect_data_counted, not_in_dict_counted, goods_not_counted, dict_incorrect, array_not_found, array_goods_not)
    
def save_ocr_values(output_file_path, values, incorrect_data, not_found_data, good_not_found):
    correctness = values[0]
    correct_data_counted = values[1]
    incorrect_data_counted = values[2]
    not_data_found_counted = values[3]
    good_not_found_counted = values[4]
    time_diff = values[5]
    
    with codecs.open(output_file_path, "+a", "utf-8") as file:
        file.write(f"{correctness};{correct_data_counted};{incorrect_data_counted};{not_data_found_counted};{good_not_found_counted};{time_diff};{incorrect_data};{not_found_data};{good_not_found}\n")

def save_object_values(output_file_path, map, map_50, map_75, map_large, mar_100, mar_large):
    with open(output_file_path, "+a") as file:
        file.write(f"{map};{map_50};{map_75};{map_large};{mar_100};{mar_large}\n")

def save_object_main_values(output_file_path, time_diff, json_loaded):
    with open(output_file_path, "+a") as file:
        file.write(f"{time_diff};{json_loaded}\n")

"""
* Save the characteristics of model response to the file.

Input: (model name, type of data, charakteristics of data and time of run, incorrect data dict, 
        not founded data array, not founded goods)
Output: None
"""  
def save_to_file_ocr(model, type_of_data, values, incorrect_data, not_found_data, good_not_found):
    create_dir_if_not_exists(test_dir_path_output)
    output_file_path = os.path.join(test_dir_path_output, f"{model}_{type_of_data}.txt")
    save_ocr_values(output_file_path, values, incorrect_data, not_found_data, good_not_found)

def save_to_file_ocr_pattern_test(model, type_of_data, values, incorrect_data, not_found_data, good_not_found, pattern_key):
    create_dir_if_not_exists(pattern_test_dir_output_path)
    output_dir_path = os.path.join(pattern_test_dir_output_path, pattern_key)
    create_dir_if_not_exists(output_dir_path)
    output_file_path = os.path.join(output_dir_path, f"{model}_{type_of_data}.txt")
    save_ocr_values(output_file_path, values, incorrect_data, not_found_data, good_not_found)

def save_to_file_object(model, type_of_data, map, map_50, map_75, map_large, mar_100, mar_large, iou):
    create_dir_if_not_exists(test_dir_objects_path_output)
    output_file_path = os.path.join(test_dir_objects_path_output, f"{model}_{type_of_data}_{iou}.txt")
    save_object_values(output_file_path, map, map_50, map_75, map_large, mar_100, mar_large)

def save_to_file_object_pattern_test(model, type_of_data, map, map_50, map_75, map_large, mar_100, mar_large, iou, pattern_key):
    create_dir_if_not_exists(pattern_test_object_dir_output_path)
    output_dir_path = os.path.join(pattern_test_object_dir_output_path, pattern_key)
    create_dir_if_not_exists(output_dir_path)
    output_file_path = os.path.join(output_dir_path, f"{model}_{type_of_data}_{iou}.txt")
    save_object_values(output_file_path, map, map_50, map_75, map_large, mar_100, mar_large)

def save_to_file_object_main(model, type_of_data, time_diff, json_loaded):
    create_dir_if_not_exists(test_dir_objects_path_output)
    output_file_path = os.path.join(test_dir_objects_path_output, f"{model}_{type_of_data}_main.txt")

    save_object_main_values(output_file_path, time_diff, json_loaded)

def save_to_file_object_main_pattern_test(model, type_of_data, time_diff, json_loaded, pattern_key):
    create_dir_if_not_exists(pattern_test_object_dir_output_path)
    output_dir_path = os.path.join(pattern_test_object_dir_output_path, pattern_key)
    create_dir_if_not_exists(output_dir_path)
    output_file_path = os.path.join(output_dir_path, f"{model}_{type_of_data}_main.txt")

    save_object_main_values(output_file_path, time_diff, json_loaded)
"""
* Save the CPU and GPU measurement to the file.

Input: (model name, are test or train data, CPU usage, RAM usage, GPU usage, VRAM usage)
Output: None
"""  
def save_to_file_cpu_gpu(model, type_of_data, is_test, cpu_usage, cpu_percentage, ram_usage, gpu_usage, vram_usage, datetime_diff):
    if is_test:
        output_file_path = f"./test_measurement/{model}_{type_of_data}.txt"
    else:
        output_file_path = f"./train_measurement/{model}_{type_of_data}.txt"
    
    with open(output_file_path, "+a") as file:
        if is_test:
            file.write(f"{cpu_usage};{cpu_percentage};{ram_usage};{gpu_usage};{vram_usage}\n")
        else:
            file.write(f"{cpu_usage};{cpu_percentage};{ram_usage};{gpu_usage};{vram_usage};{datetime_diff}\n")

def rename_model_for_save(model_name):
    if model_name == "knoopx/mobile-vlm:3b-fp16":
        return "knoopx-mobile-vlm-3b-fp16"
    elif model_name == "llava:13b":
        return "llava-13b"
    elif model_name == "llava:34b":
        return "llava-34b"
    elif model_name == "gemma3:27b":
        return "gemma3-27b"
    elif model_name == "gemma3:12b":
        return "gemma3-12b"
    elif model_name == "mistral-small3.2:24b":
        return "mistral-small3.2-24b"
    elif model_name == "gemma3:4b":
        return "gemma3-4b"
    else:
        return model_name
