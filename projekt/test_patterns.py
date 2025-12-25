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

pattern1G_OcrEn = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find the fax number show it too as fax_number. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price."
pattern1G_OcrCz = "Zjisti mi ze snímku seznam zboží. Vrať mi adresu, datum, čas a název společnosti. Když najdeš telefonní číslo, vrať ho také. Když najdeš faxové číslo, vrať ho také jako fax_number. Když najdeš číslo stolu, informace o počtu hostů nebo číslo objednávky, vrať je také. Výstup mi ukaž jako JSON. Název společnosti navrať pod klíčem company, adresu společnosti pod klíčem address, telefonní číslo pod klíčem phone_number, jméno číšníka pod klíčem server, číslo stanice pod klíčem station, číslo objednávky pod klíčem order_number, informace o stole pod klíčem table, počet hostů pod klíčem guests, mezisoučet ceny pod klíčem sub_total, daň pod klíčem tax, celkovou cenu pod klíčem total, datum pod klíčem date, čas pod klíčem time. Každý název zboží bude jako klíč JSON v klíči goods a hodnota zboží bude další JSON s množstvím zboží v klíči amount a cenou zboží v klíči price."
pattern1G_ObjectEn = "Detect all peaple. Every person is described by one JSON. Every person has the label person and with confidence score in key confidence."
pattern1G_ObjectCz = "Detekuj všechny osoby. Každá osoba je popsána jedním JSONem. Každá osoba má štítek person a s hodnotou spolehlivosti v klíči confidence."

pattern2Op_OcrEn = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price."
pattern2Op_OcrCz = "Zjisti mi ze snímku seznam zboží. Vrať mi adresu, datum, čas a název společnosti. Když najdeš telefonní číslo, vrať ho také. Když najdeš číslo stolu, informace o počtu hostů nebo číslo objednávky, vrať je také. Výstup mi ukaž jako JSON. Název společnosti navrať pod klíčem company, adresu společnosti pod klíčem address, telefonní číslo pod klíčem phone_number, jméno číšníka pod klíčem server, číslo stanice pod klíčem station, číslo objednávky pod klíčem order_number, informace o stole pod klíčem table, počet hostů pod klíčem guests, mezisoučet ceny pod klíčem sub_total, daň pod klíčem tax, celkovou cenu pod klíčem total, datum pod klíčem date, čas pod klíčem time. Každý název zboží bude jako klíč JSON v klíči goods a hodnota zboží bude další JSON s množstvím zboží v klíči amount a cenou zboží v klíči price."
pattern2Op_ObjectEn = "Detect all peaple. For every person add the string person to name key, x-min coordinate of person add to x_min key, x-max coordinate of person add to x_max key, y-min coordinate of person add to y_min key, y-max coordinate of person add to y_max key and confidence score in key confidence. All this data are given as JSON and added to JSON array. This JSON array add to key objects."
pattern2Op_ObjectCz = "Detekuj všechny osoby. Pro každou osobu přidej řetězec person do klíče name, x-min souřadnici osoby přidej do klíče x_min, x-max souřadnici osoby přidej do klíče x_max, y-min souřadnici osoby přidej do klíče y_min, y-max souřadnici osoby přidej do klíče y_max a hodnotu spolehlivosti v klíči confidence. Všechny tyto údaje jsou uvedeny jako JSON a přidány do JSON pole. Právě vytvořené JSON pole přidej do výsledku s klíčem objects."

pattern3Ol_OcrEn = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, fax number in key fax_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price. Return it as only JSON."
pattern3Ol_OcrCz = "Zjisti mi ze snímku seznam zboží. Vrať mi adresu, datum, čas a název společnosti. Když najdeš telefonní číslo, vrať ho také. Když najdeš faxové číslo, vrať ho také jako fax_number. Když najdeš číslo stolu, informace o počtu hostů nebo číslo objednávky, vrať je také. Výstup mi ukaž jako JSON. Název společnosti navrať pod klíčem company, adresu společnosti pod klíčem address, telefonní číslo pod klíčem phone_number, faxové číslo pod klíčem fax_number, jméno číšníka pod klíčem server, číslo stanice pod klíčem station, číslo objednávky pod klíčem order_number, informace o stole pod klíčem table, počet hostů pod klíčem guests, mezisoučet ceny pod klíčem sub_total, daň pod klíčem tax, celkovou cenu pod klíčem total, datum pod klíčem date, čas pod klíčem time. Každý název zboží bude jako klíč JSON v klíči goods a hodnota zboží bude další JSON s množstvím zboží v klíči amount a cenou zboží v klíči price. Vrať to pouze jako JSON."
pattern3Ol_ObjectEn = "Detect all peaple. For every person add the string person to name key, x-min coordinate of person add to x_min key, x-max coordinate of person add to x_max key, y-min coordinate of person add to y_min key, y-max coordinate of person add to y_max key and confidence score add to confidence key. All this data are given as JSON and added to JSON array. This JSON array add to key objects. Return only JSON output with detections."
pattern3Ol_ObjectCz = "Detekuj všechny osoby. Pro každou osobu přidej řetězec person do klíče name, x-min souřadnici osoby přidej do klíče x_min, x-max souřadnici osoby přidej do klíče x_max, y-min souřadnici osoby přidej do klíče y_min, y-max souřadnici osoby přidej do klíče y_max a hodnotu spolehlivosti přidej do klíče confidence. Všechny tyto údaje jsou uvedeny jako JSON a přidány do JSON pole. Právě vytvořené JSON pole přidej do výsledku s klíčem objects. Vrať to pouze jako JSON výstup s detekcemi."

pattern4Ol_OcrEn = "Get me the list of goods from picture. Show the address, date, time and name of company. When you find the phone number show it too. When you find you find the table number, the information about guest or order number show it too. Show me the output as JSON. The company name put in key company, the address of company in key address, phone number in key phone_number, fax number in key fax_number, server name in key server, station number in key station, order number in key order_number, table info in key table, number of guests in key guests, subtotal price to key sub_total, tax in key tax, total cost in key total, date in key date, time in key time. Every good name will be as key of the JSON in key goods and value of the good will be the another JSON with amount of goods in key amount and the cost of the good in key price."
pattern4Ol_OcrCz = "Zjisti mi ze snímku seznam zboží. Vrať mi adresu, datum, čas a název společnosti. Když najdeš telefonní číslo, vrať ho také. Když najdeš faxové číslo, vrať ho také jako fax_number. Když najdeš číslo stolu, informace o počtu hostů nebo číslo objednávky, vrať je také. Výstup mi ukaž jako JSON. Název společnosti navrať pod klíčem company, adresu společnosti pod klíčem address, telefonní číslo pod klíčem phone_number, faxové číslo pod klíčem fax_number, jméno číšníka pod klíčem server, číslo stanice pod klíčem station, číslo objednávky pod klíčem order_number, informace o stole pod klíčem table, počet hostů pod klíčem guests, mezisoučet ceny pod klíčem sub_total, daň pod klíčem tax, celkovou cenu pod klíčem total, datum pod klíčem date, čas pod klíčem time. Každý název zboží bude jako klíč JSON v klíči goods a hodnota zboží bude další JSON s množstvím zboží v klíči amount a cenou zboží v klíči price."
#pattern4Ol_ObjectEn - stejné jako u pattern2Op_ObjectEn
#pattern4Ol_ObjectCz - stejné jako u pattern2Op_ObjectCz

patternsOcrEn = {
    "pattern1_OcrEn": pattern1G_OcrEn,
    "pattern2_OcrEn": pattern2Op_OcrEn,
    "pattern3_OcrEn": pattern3Ol_OcrEn,
    "pattern4_OcrEn": pattern4Ol_OcrEn
}
patternsOcrCz = {
    "pattern1_OcrCz": pattern1G_OcrCz,
    "pattern2_OcrCz": pattern2Op_OcrCz,
    "pattern3_OcrCz": pattern3Ol_OcrCz,
    "pattern4_OcrCz": pattern4Ol_OcrCz
}
patternsObjectEn = {
    "pattern1_ObjectEn": pattern1G_ObjectEn,
    "pattern2_ObjectEn": pattern2Op_ObjectEn,
    "pattern3_ObjectEn": pattern3Ol_ObjectEn
}
patternsObjectCz = {
    "pattern1_ObjectCz": pattern1G_ObjectCz,
    "pattern2_ObjectCz": pattern2Op_ObjectCz,
    "pattern3_ObjectCz": pattern3Ol_ObjectCz
}

gemini_measurement = True
openai_measurement = True
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

    if type_of_data == "ticket":
        data_tuple = functions.check_the_data_ocr(response, file_name, correct_data_path, True)
        functions.save_to_file_ocr_pattern_test(model, type_of_data, [data_tuple[0], data_tuple[1], data_tuple[2], data_tuple[3],
                                                         data_tuple[4], diff_datetime_seconds], data_tuple[5],
                                                         data_tuple[6], data_tuple[7], pattern_key)
    else:
        json_response, json_loaded = functions.load_json_response(response)
        max_iou_detections, good_boxes = functions.get_max_iou_and_good_boxes(file_name, json_response["objects"])
        for iou_threshold in functions.iou_thresholds:
            map_values = functions.get_mAP(max_iou_detections, good_boxes, iou_threshold)
            functions.save_to_file_object_pattern_test(model, type_of_data, map_values["map"],
                                                       map_values["map_50"], map_values["map_75"],
                                                       map_values["map_large"], map_values["mar_100"],
                                                       map_values["mar_large"], iou_threshold, pattern_key)
        functions.save_to_file_object_main_pattern_test(model, type_of_data, diff_datetime_seconds, json_loaded, pattern_key)

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
    test_ocr()
    test_object()
