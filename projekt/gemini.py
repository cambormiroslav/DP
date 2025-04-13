import google.generativeai as genai
import os
import datetime
import time

import functions

api_key = os.environ["GEMINI_API_KEY"]

ocr_method = True

if ocr_method:
    pattern = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price."
    type_of_data = "ticket"
    correct_data_path = "../data_for_control/dataset_correct_data.json"
else:
    pattern = "What type of objekt is on this image? Return it as JSON. Type of objekt put in the key type."
    type_of_data = "objects"
    correct_data_path = "../data_for_control/dataset_objects_correct_data.json"

model_text = "gemini-1.5-flash"
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
            functions.save_to_file_ocr(model_text, type_of_data, [correctness, correct_data, 
                                                                  incorect_data, not_found_data, 
                                                                  good_not_found, diff_datetime_seconds], 
                                                                  dict_of_incorect, array_not_found, 
                                                                  array_good_not_found)
        else:
            functions.save_to_file_object(model_text, type_of_data, [correctness, correct_data,
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

        if model_is_pro:
            time.sleep(15.0)

if __name__ == "__main__":
    if ocr_method:
        dir_path = "../dataset/large-receipt-image-dataset-SRD/"
    else:
        dir_path = "../dataset/objects/"

    load_and_measure(dir_path, 1, 103)

    model_text = "gemini-1.5-pro"
    model_is_pro = True
    load_and_measure(dir_path, 1, 103)

    model_text = "gemini-2.0-flash"
    model_is_pro = False
    load_and_measure(dir_path, 1, 103)

    model_text = "gemini-2.0-flash-lite"
    model_is_pro = False
    load_and_measure(dir_path, 1, 103)

    model_text = "gemini-2.5-pro-preview-03-25"
    model_is_pro = True
    load_and_measure(dir_path, 1, 103)