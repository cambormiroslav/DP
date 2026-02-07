import os
import datetime
import shutil

import gemini
import ollama_api
import openai
import functions

delete_dirs = False

if delete_dirs:
    if os.path.exists(functions.pattern_test_dir_output_path):
        shutil.rmtree(functions.pattern_test_dir_output_path)
    if os.path.exists(functions.pattern_test_object_dir_output_path):
        shutil.rmtree(functions.pattern_test_object_dir_output_path)

patternsOcrEn = {
    "pattern1_OcrEn": functions.pattern1_OcrEn,
    "pattern2_OcrEn": functions.pattern2_OcrEn,
    "pattern3_OcrEn": functions.pattern3_OcrEn,
    "pattern4_OcrEn": functions.pattern4_OcrEn
}
patternsOcrCz = {
    "pattern1_OcrCz": functions.pattern1_OcrCz,
    "pattern2_OcrCz": functions.pattern2_OcrCz,
    "pattern3_OcrCz": functions.pattern3_OcrCz,
    "pattern4_OcrCz": functions.pattern4_OcrCz
}
patternsObjectEn = {
    "pattern1_ObjectEn": functions.pattern1_ObjectEn,
    "pattern2_ObjectEn": functions.pattern2_ObjectEn,
    "pattern3_ObjectEn": functions.pattern3_ObjectEn
}
patternsObjectCz = {
    "pattern1_ObjectCz": functions.pattern1_ObjectCz,
    "pattern2_ObjectCz": functions.pattern2_ObjectCz,
    "pattern3_ObjectCz": functions.pattern3_ObjectCz
}

gemini_measurement = False
openai_measurement = False
ollama_measurement = True

gemini_models = ["gemini-3-pro-preview", "gemini-3-flash-preview",
                 "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
                 "gemini-2.0-flash", "gemini-2.0-flash-lite"]
openai_models = ["gpt-5.2", "gpt-5.1",
                 "gpt-5", "gpt-5-mini", "gpt-5-nano", 
                 "gpt-4.1", "gpt-4.1-mini", 
                 "gpt-4o", "gpt-4o-mini"]
ollama_models = ["llava", "bakllava", "minicpm-v", "knoopx/mobile-vlm:3b-fp16", "llava:13b", "llava:34b", 
                 "gemma3:27b", "granite3.2-vision", "gemma3:12b", "gemma3:4b", "mistral-small3.1", 
                 "mistral-small3.2:24b"]

number_of_inputs = 20

def calcute_timediff_and_save(response, start_datetime, end_datetime, model, pattern_key, file_name, type_of_data, correct_data_path):
    diff_datetime = end_datetime - start_datetime
    diff_datetime_seconds = diff_datetime.total_seconds()

    renamed_model = functions.rename_model_for_save(model)

    if type_of_data == "ticket":
        data_tuple = functions.check_the_data_ocr(response, file_name, correct_data_path, True)
        functions.save_to_file_ocr_pattern_test(renamed_model, type_of_data, [data_tuple[0], data_tuple[1], data_tuple[2], data_tuple[3],
                                                         data_tuple[4], diff_datetime_seconds], data_tuple[5],
                                                         data_tuple[6], data_tuple[7], pattern_key)
    else:
        print(renamed_model)
        json_response, json_loaded = functions.load_json_response(response)
        max_iou_detections, good_boxes = functions.get_max_iou_and_good_boxes(file_name, json_response["objects"])
        for iou_threshold in functions.iou_thresholds:
            map_values = functions.get_mAP(max_iou_detections, good_boxes, iou_threshold)
            functions.save_to_file_object_pattern_test(renamed_model, type_of_data, map_values["map"],
                                                       map_values["map_50"], map_values["map_75"],
                                                       map_values["map_large"], map_values["mar_100"],
                                                       map_values["mar_large"], iou_threshold, pattern_key)
        functions.save_to_file_object_main_pattern_test(renamed_model, type_of_data, diff_datetime_seconds, json_loaded, pattern_key)

def send_gemini_request(image_path, file_name, model, text_request, pattern_key, correct_data_path, type_of_data):
    start_datetime = datetime.datetime.now()
    response = gemini.send_image_request(image_path, model, text_request)
    end_datetime = datetime.datetime.now()

    calcute_timediff_and_save(response, start_datetime, end_datetime, model, pattern_key, file_name, type_of_data, correct_data_path)

def send_openai_request(image_path, file_name, model, text_request, pattern_key, correct_data_path, type_of_data):
    start_datetime = datetime.datetime.now()
    response = openai.send_image_request(image_path, model, text_request)
    end_datetime = datetime.datetime.now()

    calcute_timediff_and_save(response, start_datetime, end_datetime, model, pattern_key, file_name, type_of_data, correct_data_path)

def send_ollama_request(image_path, file_name, model, text_request, pattern_key, correct_data_path, type_of_data):
    start_datetime = datetime.datetime.now()
    response = ollama_api.send_image_request(ollama_api.get_image_in_base64(image_path), model, text_request)
    end_datetime = datetime.datetime.now()

    calcute_timediff_and_save(response, start_datetime, end_datetime, model, pattern_key, file_name, type_of_data, correct_data_path)

def test_ocr():
    correct_data_path = "../data_for_control/dataset_correct_data.json"
    dataset_dir_path = "../dataset/large-receipt-image-dataset-SRD/"
    sorted_array_of_images = sorted(os.listdir(dataset_dir_path))
    for index in range(number_of_inputs):
        file = sorted_array_of_images[index]
        image_path = os.path.join(dataset_dir_path, file)

        if gemini_measurement:
            for model in gemini_models:
                for pattern_en in patternsOcrEn:
                    send_gemini_request(image_path, file, model, patternsOcrEn[pattern_en], pattern_en, correct_data_path, "ticket")
                for pattern_cz in patternsOcrCz:
                    send_gemini_request(image_path, file, model, patternsOcrCz[pattern_cz], pattern_cz, correct_data_path, "ticket")

        if openai_measurement:           
            for model in openai_models:
                for pattern_en in patternsOcrEn:
                    send_openai_request(image_path, file, model, patternsOcrEn[pattern_en], pattern_en, correct_data_path, "ticket")
                for pattern_cz in patternsOcrCz:
                    send_openai_request(image_path, file, model, patternsOcrCz[pattern_cz], pattern_cz, correct_data_path, "ticket")

        if ollama_measurement:            
            for model in ollama_models:
                for pattern_en in patternsOcrEn:
                    send_ollama_request(image_path, file, model, patternsOcrEn[pattern_en], pattern_en, correct_data_path, "ticket")
                for pattern_cz in patternsOcrCz:
                    send_ollama_request(image_path, file, model, patternsOcrCz[pattern_cz], pattern_cz, correct_data_path, "ticket")

def test_object():
    correct_data_path = "../data_for_control/dataset_objects_correct_data.json"
    dataset_dir_path = "../dataset/objects/"
    sorted_array_of_images = sorted(os.listdir(dataset_dir_path))
    for index in range(number_of_inputs):
        file = sorted_array_of_images[index]
        image_path = os.path.join(dataset_dir_path, file)

        if gemini_measurement:
            for model in gemini_models:
                for pattern_en in patternsObjectEn:
                    send_gemini_request(image_path, file, model, patternsObjectEn[pattern_en], pattern_en, correct_data_path, "object")
                for pattern_cz in patternsObjectCz:
                    send_gemini_request(image_path, file, model, patternsObjectCz[pattern_cz], pattern_cz, correct_data_path, "object")
        
        if openai_measurement:
            for model in openai_models:
                for pattern_en in patternsObjectEn:
                    send_openai_request(image_path, file, model, patternsObjectEn[pattern_en], pattern_en, correct_data_path, "object")
                for pattern_cz in patternsObjectCz:
                    send_openai_request(image_path, file, model, patternsObjectCz[pattern_cz], pattern_cz, correct_data_path, "object")
        
        if ollama_measurement:
            for model in ollama_models:
                for pattern_en in patternsObjectEn:
                    send_ollama_request(image_path, file, model, patternsObjectEn[pattern_en], pattern_en, correct_data_path, "object")
                for pattern_cz in patternsObjectCz:
                    send_ollama_request(image_path, file, model, patternsObjectCz[pattern_cz], pattern_cz, correct_data_path, "object")

if __name__ == "__main__":
    #test_ocr()
    test_object()
