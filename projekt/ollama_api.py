import requests
import json
import base64
import os
import time

import functions

pattern = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price."
correct_data_path = "../data_for_control/dataset_correct_data.json"

model = "llava"
#model = "bakllava"
#model = "minicpm-v"
#model  = "knoopx/mobile-vlm:3b-fp16"


def get_image_in_base64(path_to_image):
    with open(path_to_image, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    return image_base64

def send_image_request(image_base64, text_request):
    url_llava = "http://localhost:11434/api/generate"
    payload = {
        "model" : model,
        "prompt" : text_request,
        "stream" : False,
        "images" : [image_base64]
    }

    response = requests.post(url_llava, data=json.dumps(payload))

    return response.json()["response"].replace("```json\n", "").replace("\n```", "")

def send_image_request_all(image_base64, text_request):
    url_llava = "http://localhost:11434/api/generate"
    payload = {
        "model" : model,
        "prompt" : text_request,
        "stream" : False,
        "images" : [image_base64]
    }

    response = requests.post(url_llava, data=json.dumps(payload))

    return response.json()

def get_the_data_from_img_in_dir(dir_path):
    i = 0
    array_of_outputs = []
    array_of_diffs = []
    for file in os.listdir(dir_path):
        start_time = time.time()
        base_64_image = get_image_in_base64(dir_path + file)
        response = send_image_request(base_64_image, pattern)
        end_time = time.time()
        diff_time = end_time - start_time
        array_of_outputs += [response]
        array_of_diffs += [diff_time]
        i += 1
        print(i)
    return (array_of_outputs, array_of_diffs)
    
if __name__ == "__main__":
    dir_path = "../dataset/large-receipt-image-dataset-SRD/"

    """ dir_path = "../dataset/large-receipt-image-dataset-SRD/"
    array_of_outputs, array_of_diffs = get_the_data_from_img_in_dir(dir_path)
    print(array_of_outputs[2], len(array_of_outputs), array_of_diffs[2])
    print(functions.get_avg_time_run(array_of_diffs)) """

    #print(send_image_request_all(get_image_in_base64(dir_path + "1000-receipt.jpg"), pattern))

    correctness, correct_data, incorect_data, not_found_data, dict_of_incorect, array_not_found = functions.check_the_data(json.loads(send_image_request(get_image_in_base64(dir_path + "1000-receipt.jpg"), pattern)), "1000-receipt.jpg", correct_data_path)
    print(correctness, correct_data, incorect_data, not_found_data, dict_of_incorect, array_not_found)
    functions.save_to_file(model, "ticket", [correctness, correct_data, incorect_data, not_found_data], dict_of_incorect, array_not_found)