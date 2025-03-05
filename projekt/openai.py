import base64
import requests
import os
import datetime

import functions

api_key = os.environ["OPENAI_API_KEY"]
pattern = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price."
correct_data_path = "../data_for_control/dataset_correct_data.json"

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
        "model": "gpt-4o-mini",
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
def load_and_measure(dir_path, number_of_tickets):
    i = 0
    for file in os.listdir(dir_path):
        start_datetime = datetime.datetime.now()
        response = send_image_request(dir_path + file, pattern)
        end_datetime = datetime.datetime.now()

        data_tuple = functions.check_the_data(response, file, correct_data_path)
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

        functions.save_to_file("chatgpt", "ticket", [correctness, correct_data, 
                                                 incorect_data, not_found_data, 
                                                 good_not_found, diff_datetime_seconds], 
                                                 dict_of_incorect, array_not_found, 
                                                 array_good_not_found)
        print(correctness, correct_data, incorect_data, not_found_data, 
              good_not_found, diff_datetime_seconds, dict_of_incorect,
              array_not_found, array_good_not_found)
        i += 1
        print("Receipt: ", i)

        if i == number_of_tickets:
            break

if __name__ == "__main__":
    dir_path = "../dataset/large-receipt-image-dataset-SRD/"
    
    load_and_measure(dir_path, 3)

    """ correctness, correct_data, incorect_data, not_found_data, dict_of_incorect, array_not_found = functions.check_the_data(send_image_request(dir_path + "1000-receipt.jpg", pattern), "1000-receipt.jpg", correct_data_path)
    print(correctness, correct_data, incorect_data, not_found_data, dict_of_incorect, array_not_found)
    functions.save_to_file("openai", "ticket", [correctness, correct_data, incorect_data, not_found_data], dict_of_incorect, array_not_found) """