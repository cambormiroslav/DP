import requests
import json
import base64
import os
import datetime

import functions

ocr_method = True

if ocr_method:
    pattern = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, fax number in key fax_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price."
    correct_data_path = "../data_for_control/dataset_correct_data.json"
else:
    pattern = "What type of the animal is in the photography and breed of this animal? Return it as JSON type of animal put in the key type and animal breed in key breed."
    correct_data_path = "../data_for_control/dataset_images_correct_data.json"


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
        start_datetime = datetime.datetime.now()
        base_64_image = get_image_in_base64(dir_path + file)
        response = send_image_request(base_64_image, pattern)
        end_datetime = datetime.datetime.now()

        if ocr_method:
            data_tuple = functions.check_the_data_ocr(response, file, correct_data_path)
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
                functions.save_to_file_ocr("knoopx-mobile-vlm-3b-fp16", "ticket", [correctness, correct_data, 
                                                                                   incorect_data, not_found_data, 
                                                                                   good_not_found, diff_datetime_seconds], 
                                                                                   dict_of_incorect, array_not_found, 
                                                                                   array_good_not_found)
            else:
                functions.save_to_file_object("knoopx-mobile-vlm-3b-fp16", "animals", [correctness, correct_data, 
                                                                                   incorect_data, not_found_data, diff_datetime_seconds], 
                                                                                   dict_of_incorect, array_not_found)
        elif model == "llava:13b":
            if ocr_method:
                functions.save_to_file_ocr("llava-13b", "ticket", [correctness, correct_data, 
                                                                   incorect_data, not_found_data, 
                                                                   good_not_found, diff_datetime_seconds], 
                                                                   dict_of_incorect, array_not_found,
                                                                   array_good_not_found)
            else:
                functions.save_to_file_object("llava-13b", "animals", [correctness, correct_data,
                                                                    incorect_data, not_found_data, diff_datetime_seconds],
                                                                    dict_of_incorect, array_not_found)
        elif model == "llava:34b":
            if ocr_method:
                functions.save_to_file_ocr("llava-34b", "ticket", [correctness, correct_data, 
                                                                   incorect_data, not_found_data, 
                                                                   good_not_found, diff_datetime_seconds], 
                                                                   dict_of_incorect, array_not_found, 
                                                                   array_good_not_found)
            else:
                functions.save_to_file_object("llava-34b", "animals", [correctness, correct_data,
                                                                    incorect_data, not_found_data, diff_datetime_seconds],
                                                                    dict_of_incorect, array_not_found)
        elif model == "gemma3:27b":
            if ocr_method:
                functions.save_to_file_ocr("gemma3-27b", "ticket", [correctness, correct_data, 
                                                                   incorect_data, not_found_data, 
                                                                   good_not_found, diff_datetime_seconds], 
                                                                   dict_of_incorect, array_not_found, 
                                                                   array_good_not_found)
            else:
                functions.save_to_file_object("gemma3-27b", "animals", [correctness, correct_data,
                                                                    incorect_data, not_found_data, diff_datetime_seconds],
                                                                    dict_of_incorect, array_not_found)
        elif model == "llama3.2-vision:11b":
            if ocr_method:
                functions.save_to_file_ocr("llama3.2-vision-11b", "ticket", [correctness, correct_data, 
                                                                   incorect_data, not_found_data, 
                                                                   good_not_found, diff_datetime_seconds], 
                                                                   dict_of_incorect, array_not_found, 
                                                                   array_good_not_found)
            else:
                functions.save_to_file_object("llama3.2-vision-11b", "animals", [correctness, correct_data,
                                                                    incorect_data, not_found_data, diff_datetime_seconds],
                                                                    dict_of_incorect, array_not_found)
        else:
            if ocr_method:
                functions.save_to_file_ocr(model, "ticket", [correctness, correct_data, 
                                                             incorect_data, not_found_data, 
                                                             good_not_found, diff_datetime_seconds], 
                                                             dict_of_incorect, array_not_found,
                                                             array_good_not_found)
            else:
                functions.save_to_file_object(model, "animals", [correctness, correct_data,
                                                                    incorect_data, not_found_data, diff_datetime_seconds],
                                                                    dict_of_incorect, array_not_found)
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
        dir_path = "../dataset/images_dataset/"

    #load_and_measure(dir_path, 1, 103)
    #load_and_measure(dir_path, 1, 32)
    #print(send_image_request(get_image_in_base64(dir_path + "1000-receipt.jpg"), pattern))
    
    model = "bakllava"
    #load_and_measure(dir_path, 1, 103)
    #print(send_image_request(get_image_in_base64(dir_path + "1000-receipt.jpg"), pattern))

    model = "minicpm-v"
    #load_and_measure(dir_path, 1, 103)
    #print(send_image_request(get_image_in_base64(dir_path + "1000-receipt.jpg"), pattern))

    model = "knoopx/mobile-vlm:3b-fp16"
    #load_and_measure(dir_path, 1, 103)
    #print(send_image_request(get_image_in_base64(dir_path + "1000-receipt.jpg"), pattern))

    model = "llava:13b"
    #load_and_measure(dir_path, 1, 103)
    #print(send_image_request(get_image_in_base64(dir_path + "1000-receipt.jpg"), pattern))

    model = "llava:34b"
    #load_and_measure(dir_path, 1, 103)
    #print(send_image_request(get_image_in_base64(dir_path + "1000-receipt.jpg"), pattern))

    model = "gemma3:27b"
    #load_and_measure(dir_path, 21, 103)

    model = "granite3.2-vision"
    #load_and_measure(dir_path, 21, 103)

    model = "mistral-small3.1"
    #load_and_measure(dir_path, 67, 103)

    model = "gemma3:12b"
    load_and_measure(dir_path, 1, 103)

    model = "gemma3:4b"
    load_and_measure(dir_path, 1, 103)

    model = "gemma3:1b"
    load_and_measure(dir_path, 1, 103)