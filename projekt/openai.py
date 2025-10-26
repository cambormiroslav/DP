import base64
import requests
import os
import datetime
import psutil
import threading

import functions

api_key = os.environ["OPENAI_API_KEY"]

ocr_method = False

if ocr_method:
    pattern = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find the fax number show it too as fax_number. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price."
    type_of_data = "ticket"
    correct_data_path = "../data_for_control/dataset_correct_data.json"
else:
    pattern = "What type of objekt is on this image? Return it as JSON. Type of objekt put in the key type."
    type_of_data = "objects"
    correct_data_path = "../data_for_control/dataset_objects_correct_data.json"

model = "gpt-4o-mini"

"""
* Send request to the model
* Transform input image to base64.

Input: (Image in base64, Text pattern for model)
Output: Response as text
"""
def send_image_request(image_path, text_request):
    base64_image = None
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_request
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    return requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()["choices"][0]["message"]["content"].replace("Here's the extracted information in JSON format:\n\n", "").replace("```json\n", "").replace("\n```", "")

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
        #cpu and memory before test model
        process.cpu_percent(interval=None)
        mem_before = process.memory_info().rss / (1024 * 1024)

        functions.monitor_data["is_running"] = True
        monitor_thread = threading.Thread(
            target=functions.monitor_memory, 
            args=(process,),
            daemon=True #stops if main script stops
        )
        monitor_thread.start()

        start_datetime = datetime.datetime.now()

        try:
            response = send_image_request(dir_path + file, pattern)
        finally:
            # stop thread
            functions.monitor_data["is_running"] = False
            monitor_thread.join(timeout=1.0)
        
        end_datetime = datetime.datetime.now()
        #get cpu and ram usage
        mem_after = process.memory_info().rss / (1024 * 1024)
        peak_ram_mb = functions.monitor_data["peak_rss_mb"]
        cpu_usage = process.cpu_percent(interval=None)

        peak_ram_mb = max(peak_ram_mb, mem_after) #maximum of peak RAM and final value of RAM
        ram_usage = peak_ram_mb - mem_before

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
        functions.save_to_file_cpu_gpu(model, True, cpu_usage, ram_usage, diff_datetime_seconds)

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
    
    load_and_measure(dir_path, 1, 103)

    model = "gpt-4o"
    load_and_measure(dir_path, 1, 103)

    model = "gpt-4.1-nano"
    load_and_measure(dir_path, 1, 103)

    model = "gpt-4.1-mini"
    load_and_measure(dir_path, 1, 103)

    model = "gpt-4.1"
    load_and_measure(dir_path, 1, 103)

    model = "gpt-5-nano"
    load_and_measure(dir_path, 1, 103)

    model = "gpt-5-mini"
    load_and_measure(dir_path, 1, 103)

    model = "gpt-5"
    load_and_measure(dir_path, 1, 103)