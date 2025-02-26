import base64
import requests
import os
import time
import json

import functions

api_key = os.environ["OPENAI_API_KEY"]
pattern = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price."
correct_data_path = "../data_for_control/dataset_correct_data.json"

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

def get_the_data_from_img_in_dir(dir_path):
    i = 0
    array_of_outputs = []
    array_of_diffs = []
    for file in os.listdir(dir_path):
        start_time = time.time()
        response = send_image_request(dir_path + file, pattern)
        end_time = time.time()
        diff_time = end_time - start_time
        array_of_outputs += [response]
        array_of_diffs += [diff_time]
        i += 1
        print(i)

    return (array_of_outputs, array_of_diffs)

if __name__ == "__main__":
    dir_path = "../dataset/large-receipt-image-dataset-SRD/"
    
    """ array_of_outputs, array_of_diffs = get_the_data_from_img_in_dir(dir_path)
    print(array_of_outputs[2], len(array_of_outputs), array_of_diffs[2])
    print(functions.get_avg_time_run(array_of_diffs)) """

    correctness, correct_data, incorect_data, not_found_data, dict_of_incorect, array_not_found = functions.check_the_data(send_image_request(dir_path + "1000-receipt.jpg", pattern), "1000-receipt.jpg", correct_data_path)
    print(correctness, correct_data, incorect_data, not_found_data, dict_of_incorect, array_not_found)
    functions.save_to_file("openai", "ticket", [correctness, correct_data, incorect_data, not_found_data], dict_of_incorect, array_not_found)