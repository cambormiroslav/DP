import os
import xml.etree.ElementTree as ET
import json

xml_annotations_files_dir = "../data_for_control/pascal_annotations/"
output_json_path = "../data_for_control/dataset_objects_correct_data.json"

def generate_data_for_control_file():
    dict_for_control_file = {}
    sorted_annotation_files = sorted(os.listdir(xml_annotations_files_dir))
    for annotation_file in sorted_annotation_files:
        print(annotation_file)
        root = ET.parse(xml_annotations_files_dir + annotation_file)

        file_name_found = root.findall('filename')[0].text

        object_found = root.findall('object')
        for obj in object_found:
            bndbox = obj.findall('bndbox')[0]
            xmin = int(bndbox.findall('xmin')[0].text)
            ymin = int(bndbox.findall('ymin')[0].text)
            xmax = int(bndbox.findall('xmax')[0].text)
            ymax = int(bndbox.findall('ymax')[0].text)

            name = obj.findall('name')[0].text
            
            if file_name_found not in dict_for_control_file:
                dict_for_control_file[file_name_found] = [{
                    "name": name,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }]
            else:
                dict_for_control_file[file_name_found].append({
                    "name": name,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                })
    
    return dict_for_control_file

def save_data_for_control_file(data):
    with open(output_json_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    save_data_for_control_file(generate_data_for_control_file())
