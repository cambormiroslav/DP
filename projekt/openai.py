import base64
import requests
import os
import datetime

import functions

api_key = os.environ["OPENAI_API_KEY"]

ocr_method = True

if ocr_method:
    pattern = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find the fax number show it too as fax_number. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price."
    correct_data_path = "../data_for_control/dataset_correct_data.json"
else:
    pattern = "What type of the animal is in the photography and breed of this animal? Return it as JSON type of animal put in the key type and animal breed in key breed."
    correct_data_path = "../data_for_control/dataset_images_correct_data.json"

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
        start_datetime = datetime.datetime.now()
        response = send_image_request(dir_path + file, pattern)
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
        
        diff_datetime = end_datetime - start_datetime
        diff_datetime_seconds = diff_datetime.total_seconds()

        if ocr_method:
            functions.save_to_file_ocr(model, "ticket", [correctness, correct_data, 
                                                         incorect_data, not_found_data, 
                                                         good_not_found, diff_datetime_seconds], 
                                                         dict_of_incorect, array_not_found, 
                                                         array_good_not_found)
        if ocr_method:
            print(correctness, correct_data, incorect_data, not_found_data, 
                  good_not_found, diff_datetime_seconds, dict_of_incorect,
                  array_not_found, array_good_not_found)
        
        i += 1

        if ocr_method:
            print("Receipt: ", i)

        if i == latest_file:
            break

if __name__ == "__main__":
    if ocr_method:
        dir_path = "../dataset/large-receipt-image-dataset-SRD/"
    else:
        dir_path = "../dataset/images_dataset/"
    
    load_and_measure(dir_path, 1, 103)

    model = "gpt-4.5-preview"
    load_and_measure(dir_path, 1, 103)

    model = "gpt-4o"
    load_and_measure(dir_path, 1, 103)