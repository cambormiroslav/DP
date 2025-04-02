import google.generativeai as genai
import os
import datetime

import functions

api_key = os.environ["GEMINI_API_KEY"]
pattern = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price."
correct_data_path = "../data_for_control/dataset_correct_data.json"

model = "gemini-1.5-flash"

"""
* Send request to the model

Input: (Image in base64, Text pattern for model)
Output: Response as text
"""
def send_image_request(image_path, text_request):
    myfile = genai.upload_file(image_path)
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model)
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
def load_and_measure(dir_path, first_ticket, latest_ticket):
    i = first_ticket - 1
    array_of_images = os.listdir(dir_path)
    while(True):
        file = array_of_images[i]
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

        functions.save_to_file(model, "ticket", [correctness, correct_data, 
                                                 incorect_data, not_found_data, 
                                                 good_not_found, diff_datetime_seconds], 
                                                 dict_of_incorect, array_not_found, 
                                                 array_good_not_found)
        print(correctness, correct_data, incorect_data, not_found_data, 
              good_not_found, diff_datetime_seconds, dict_of_incorect,
              array_not_found, array_good_not_found)
        i += 1
        print("Receipt: ", i)

        if i == latest_ticket:
            break

if __name__ == "__main__":
    dir_path = "../dataset/large-receipt-image-dataset-SRD/"
    
    """ array_of_outputs, array_of_diffs = get_the_data_from_img_in_dir(dir_path)
    print(array_of_outputs[2], len(array_of_outputs), array_of_diffs[2])
    print(get_avg_time_run(array_of_diffs)) """

    correctness, correct_data, incorect_data, not_found_data, dict_of_incorect, array_not_found = functions.check_the_data(send_image_request(dir_path + "1000-receipt.jpg", pattern), "1000-receipt.jpg", correct_data_path)
    print(correctness, correct_data, incorect_data, not_found_data, dict_of_incorect, array_not_found)
    functions.save_to_file("gemini", "ticket", [correctness, correct_data, incorect_data, not_found_data], dict_of_incorect, array_not_found)