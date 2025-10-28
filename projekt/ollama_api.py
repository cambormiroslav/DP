import requests
import json
import base64
import os
import datetime
import psutil
import pynvml
import threading

import functions

ocr_method = True
is_mistral = True

if ocr_method:
    if is_mistral:
        pattern = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, fax number in key fax_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price. Return it as only JSON."
    else:
        pattern = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, fax number in key fax_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price."
    type_of_data = "ticket"
    correct_data_path = "../data_for_control/dataset_correct_data.json"
else:
    if is_mistral:
        pattern = "What type of objekt is on this image? Return it as only JSON. Type of objekt put in the key type."
    else:
        pattern = "What type of objekt is on this image? Return it as JSON. Type of objekt put in the key type."
    type_of_data = "objects"
    correct_data_path = "../data_for_control/dataset_objects_correct_data.json"


model = "llava"
#model = "bakllava"
#model = "minicpm-v"
#model  = "knoopx/mobile-vlm:3b-fp16"

"""
* Transform the input image to base64.

Input: Path to input image
Output: Base64 image
"""
def get_image_in_base64(path_to_image):
    with open(path_to_image, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    return image_base64

"""
* Send request to the model

Input: (Image in base64, Text pattern for model)
Output: Response as text
"""
def send_image_request(image_base64, text_request):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model" : model,
        "prompt" : text_request,
        "stream" : False,
        "images" : [image_base64]
    }

    response = requests.post(url, data=json.dumps(payload))

    return response.json()["response"].replace("```json\n", "").replace("\n```", "")

"""
* Load the specified number of images from directory path.
* Measure the time of run between request and response of model is seconds.
* Call reformat the image as base64.
* Call method that send request to model.
* Call check of data got from model as response.
* Call saving of got data from checking data for future data processing.

Input: (Path to directory with input images, count of input images)
Output: None (but call save to file)
"""
def load_and_measure(dir_path, first_ticket, latest_file):
    i = first_ticket - 1
    array_of_images = os.listdir(dir_path)
    while(True):
        file = array_of_images[i]
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
        start_datetime = datetime.datetime.now()

        try:
            base_64_image = get_image_in_base64(dir_path + file)
            response = send_image_request(base_64_image, pattern)
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

        if ocr_method:
            data_tuple = functions.check_the_data_ocr(response, file, correct_data_path, True)
            correctness = data_tuple[0]
            correct_data = data_tuple[1]
            incorect_data = data_tuple[2]
            not_found_data = data_tuple[3]
            good_not_found = data_tuple[4]
            dict_of_incorect = data_tuple[5]
            array_not_found = data_tuple[6]
            array_good_not_found = data_tuple[7]
        else:
            data_tuple = functions.check_the_data_object(response, file, correct_data_path)
            correctness = data_tuple[0]
            correct_data = data_tuple[1]
            incorect_data = data_tuple[2]
            not_found_data = data_tuple[3]
            dict_of_incorect = data_tuple[4]
            array_not_found = data_tuple[5]
        
        diff_datetime = end_datetime - start_datetime
        diff_datetime_seconds = diff_datetime.total_seconds()

        if model == "knoopx/mobile-vlm:3b-fp16":
            if ocr_method:
                functions.save_to_file_ocr("knoopx-mobile-vlm-3b-fp16", type_of_data, [correctness, correct_data, 
                                                                                   incorect_data, not_found_data, 
                                                                                   good_not_found, diff_datetime_seconds], 
                                                                                   dict_of_incorect, array_not_found, 
                                                                                   array_good_not_found)
            else:
                functions.save_to_file_object("knoopx-mobile-vlm-3b-fp16", type_of_data, [correctness, correct_data, 
                                                                                   incorect_data, not_found_data, diff_datetime_seconds], 
                                                                                   dict_of_incorect, array_not_found)
            functions.save_to_file_cpu_gpu("knoopx-mobile-vlm-3b-fp16", type_of_data, True, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                       ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb,
                                       diff_datetime_seconds)
        elif model == "llava:13b":
            if ocr_method:
                functions.save_to_file_ocr("llava-13b", type_of_data, [correctness, correct_data, 
                                                                   incorect_data, not_found_data, 
                                                                   good_not_found, diff_datetime_seconds], 
                                                                   dict_of_incorect, array_not_found,
                                                                   array_good_not_found)
            else:
                functions.save_to_file_object("llava-13b", type_of_data, [correctness, correct_data,
                                                                    incorect_data, not_found_data, diff_datetime_seconds],
                                                                    dict_of_incorect, array_not_found)
            functions.save_to_file_cpu_gpu("llava-13b", type_of_data, True, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                       ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb,
                                       diff_datetime_seconds)
        elif model == "llava:34b":
            if ocr_method:
                functions.save_to_file_ocr("llava-34b", type_of_data, [correctness, correct_data, 
                                                                   incorect_data, not_found_data, 
                                                                   good_not_found, diff_datetime_seconds], 
                                                                   dict_of_incorect, array_not_found, 
                                                                   array_good_not_found)
            else:
                functions.save_to_file_object("llava-34b", type_of_data, [correctness, correct_data,
                                                                    incorect_data, not_found_data, diff_datetime_seconds],
                                                                    dict_of_incorect, array_not_found)
            functions.save_to_file_cpu_gpu("llava-34b", type_of_data, True, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                       ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb,
                                       diff_datetime_seconds)
        elif model == "gemma3:27b":
            if ocr_method:
                functions.save_to_file_ocr("gemma3-27b", type_of_data, [correctness, correct_data, 
                                                                   incorect_data, not_found_data, 
                                                                   good_not_found, diff_datetime_seconds], 
                                                                   dict_of_incorect, array_not_found, 
                                                                   array_good_not_found)
            else:
                functions.save_to_file_object("gemma3-27b", type_of_data, [correctness, correct_data,
                                                                    incorect_data, not_found_data, diff_datetime_seconds],
                                                                    dict_of_incorect, array_not_found)
            functions.save_to_file_cpu_gpu("gemma3-27b", type_of_data, True, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                       ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb,
                                       diff_datetime_seconds)
        elif model == "gemma3:12b":
            if ocr_method:
                functions.save_to_file_ocr("gemma3-12b", type_of_data, [correctness, correct_data, 
                                                                   incorect_data, not_found_data, 
                                                                   good_not_found, diff_datetime_seconds], 
                                                                   dict_of_incorect, array_not_found, 
                                                                   array_good_not_found)
            else:
                functions.save_to_file_object("gemma3-12b", type_of_data, [correctness, correct_data,
                                                                    incorect_data, not_found_data, diff_datetime_seconds],
                                                                    dict_of_incorect, array_not_found)
            functions.save_to_file_cpu_gpu("gemma3-12b", type_of_data, True, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                       ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb,
                                       diff_datetime_seconds)
        elif model == "gemma3:4b":
            if ocr_method:
                functions.save_to_file_ocr("gemma3-4b", type_of_data, [correctness, correct_data, 
                                                                   incorect_data, not_found_data, 
                                                                   good_not_found, diff_datetime_seconds], 
                                                                   dict_of_incorect, array_not_found, 
                                                                   array_good_not_found)
            else:
                functions.save_to_file_object("gemma3-4b", type_of_data, [correctness, correct_data,
                                                                    incorect_data, not_found_data, diff_datetime_seconds],
                                                                    dict_of_incorect, array_not_found)
            functions.save_to_file_cpu_gpu("gemma3-4b", type_of_data, True, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                       ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb,
                                       diff_datetime_seconds)
        else:
            if ocr_method:
                functions.save_to_file_ocr(model, type_of_data, [correctness, correct_data, 
                                                             incorect_data, not_found_data, 
                                                             good_not_found, diff_datetime_seconds], 
                                                             dict_of_incorect, array_not_found,
                                                             array_good_not_found)
            else:
                functions.save_to_file_object(model, type_of_data, [correctness, correct_data,
                                                                    incorect_data, not_found_data, diff_datetime_seconds],
                                                                    dict_of_incorect, array_not_found)
            functions.save_to_file_cpu_gpu(model, type_of_data, True, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                       ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb,
                                       diff_datetime_seconds)

        if ocr_method:
            print(correctness, correct_data, incorect_data, not_found_data, 
                  good_not_found, diff_datetime_seconds, dict_of_incorect,
                  array_not_found, array_good_not_found)
        else:
            print(correctness, correct_data, incorect_data, not_found_data,
                  diff_datetime_seconds, dict_of_incorect, array_not_found)
        
        i += 1

        if ocr_method:
            print("Receipt: ", i)
        else:
            print("Object: ", i)

        if i == latest_file:
            break
    
if __name__ == "__main__":
    if ocr_method:
        dir_path = "../dataset/large-receipt-image-dataset-SRD/"
    else:
        dir_path = "../dataset/objects/"

    #load_and_measure(dir_path, 1, 103)
    
    model = "bakllava"
    #load_and_measure(dir_path, 1, 103)

    model = "minicpm-v"
    #load_and_measure(dir_path, 1, 103)

    model = "knoopx/mobile-vlm:3b-fp16"
    #load_and_measure(dir_path, 1, 103)

    model = "llava:13b"
    #load_and_measure(dir_path, 1, 103)

    model = "llava:34b"
    #load_and_measure(dir_path, 1, 103)

    model = "gemma3:27b"
    #load_and_measure(dir_path, 1, 103)

    model = "granite3.2-vision"
    #load_and_measure(dir_path, 1, 103)

    model = "gemma3:12b"
    #load_and_measure(dir_path, 1, 103)

    model = "gemma3:4b"
    #load_and_measure(dir_path, 1, 103)

    model = "mistral-small3.1"
    load_and_measure(dir_path, 75, 103)