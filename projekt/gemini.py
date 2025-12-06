import google.generativeai as genai
import os
import datetime
import time
import psutil
import pynvml
import threading

import functions

api_key = os.environ["GEMINI_API_KEY"]

ocr_method = False

if ocr_method:
    pattern = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price."
    type_of_data = "ticket"
    correct_data_path = "../data_for_control/dataset_correct_data.json"
else:
    pattern = "Detect all peaple. Every person is described by one JSON. Every person has the label person."
    type_of_data = "objects"
    correct_data_path = "../data_for_control/dataset_objects_correct_data.json"

model_text = "gemini-2.0-flash-lite"
model_is_pro = False

"""
* Send request to the model

Input: (Image in base64, Text pattern for model)
Output: Response as text
"""
def send_image_request(image_path, text_request):
    myfile = genai.upload_file(image_path)
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_text)
    result = model.generate_content(
        [myfile, "\n\n", text_request]
        )
    return result.text.replace("```json\n", "").replace("\n```", "")

"""
* Load the specified number of images from directory path.
* Measure the time of run between request and response of model is seconds.
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

        detections = []
        output_array = []

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
            response = send_image_request(dir_path + file, pattern)
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
            json_response = functions.load_json_response_gemini(response)
            
            for resp in json_response:
                box_coord = resp["box_2d"]
                
                detections.append({
                    "class_name": resp["label"],
                    "x_min": int(box_coord[0]),
                    "y_min": int(box_coord[1]),
                    "x_max": int(box_coord[2]),
                    "y_max": int(box_coord[3])
                })
            
            max_iou_detections, good_boxes = functions.get_max_iou_and_good_boxes(file, detections)
            for iou_threshold in functions.iou_thresholds:
                tp, fp, tn, fn, precision, recall = functions.get_tp_fp_tn_fn_precision_recall(max_iou_detections, 
                                                                                               good_boxes, iou_threshold)
                output_array.append({
                    "TP": tp,
                    "FP" : fp,
                    "TN": tn,
                    "FN" : fn,
                    "Precision": precision,
                    "Recall" : recall,
                    "IoU" : iou_threshold
                })


        
        diff_datetime = end_datetime - start_datetime
        diff_datetime_seconds = diff_datetime.total_seconds()

        if ocr_method:
            functions.save_to_file_ocr(model_text, type_of_data, [correctness, correct_data, 
                                                                  incorect_data, not_found_data, 
                                                                  good_not_found, diff_datetime_seconds], 
                                                                  dict_of_incorect, array_not_found, 
                                                                  array_good_not_found)
        else:
            for output in output_array:
                functions.save_to_file_object(model_text, type_of_data, output["TP"], output["FP"], output["TN"],
                                               output["FN"], output["Precision"], output["Recall"], diff_datetime_seconds, output["IoU"])
        functions.save_to_file_cpu_gpu(model_text, type_of_data, True, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                       ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb,
                                       diff_datetime_seconds)
        
        if ocr_method:
            print(correctness, correct_data, incorect_data, not_found_data, 
                  good_not_found, diff_datetime_seconds, dict_of_incorect,
                  array_not_found, array_good_not_found)
            
        i += 1

        if ocr_method:
            print("Receipt: ", i)
        else:
            print("Object: ", i)

        if i == latest_file:
            break

        if model_is_pro and ocr_method:
            time.sleep(15.0)
        if model_is_pro and not ocr_method:
            time.sleep(30.0)
        if ocr_method == False and model_is_pro == False:
            time.sleep(15.0)

if __name__ == "__main__":
    if ocr_method:
        dir_path = "../dataset/large-receipt-image-dataset-SRD/"
    else:
        dir_path = "../dataset/objects/"

    model_is_pro = False
    #load_and_measure(dir_path, 1, 103)

    model_text = "gemini-2.0-flash"
    model_is_pro = False
    #load_and_measure(dir_path, 1, 103)

    model_text = "gemini-2.5-flash-lite"
    model_is_pro = False
    #load_and_measure(dir_path, 1, 103)

    model_text = "gemini-2.5-flash"
    model_is_pro = False
    #load_and_measure(dir_path, 1, 103)

    model_text = "gemini-2.5-pro"
    model_is_pro = True
    load_and_measure(dir_path, 93, 103)