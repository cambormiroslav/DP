import os
import matplotlib.pyplot as plt

type_of_dataset = "ticket"
#type_of_dataset = "objects"

is_best_data = False

add_to_graph = {
    "bakllava" : True,
    "easyocr": True,
    "gemini-2.0-flash-lite": True,
    "gemini-2.0-flash" : True,
    "gemini-2.5-flash-lite": True,
    "gemini-2.5-flash" : True,
    "gemini-2.5-pro" : True,
    "gemma3-4b" : True,
    "gemma3-12b": True,
    "gemma3-27b" : True,
    "gpt-4o-mini" : True,
    "gpt-4o" : True,
    "gpt-4.1-nano" : True,
    "gpt-4.1-mini" : True,
    "gpt-4.1" : True,
    "gpt-5-nano" : True,
    "gpt-5-mini" : True,
    "gpt-5" : True,
    "granite3.2-vision" : True,
    "knoopx-mobile-vlm-3b-fp16" : True,
    "llava-7b" : True,
    "llava-13b" : True,
    "llava-34b" : True,
    "minicpm-v": True,
    "mistral-small3.1" : True,
    "tesseract-5.3.0" : True,
    "yolo11n" : True,
    "yolo11s" : True,
    "yolo11m" : True,
    "yolo11l" : True,
    "yolo11x" : True
    }

graphs_dir = "./graphs/"
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)

if type_of_dataset == "ticket":
    output_dir = "./output/"
else:
    output_dir = "./output_objects/"

correctness_dict = {}
correct_data_count_dict = {}
incorrect_data_count_dict = {}
not_finded_main_count_key_dict = {}
goods_not_finded_count_dict = {}
time_run_dict = {}
not_found_json_dict = {}
precision_sum_dict = {}
recall_sum_dict = {}

def get_count_of_all_data(correct_data, incorect_data, not_finded, goods_not_finded):
    count_of_all_data = correct_data + incorect_data + not_finded + 3 * goods_not_finded
    return count_of_all_data

def generate_boxplot(tick_labels, values, y_label, type_data):
    colors = ['blue', 'green', 'red', 'purple', 'brown',
              'pink', 'gray', 'olive', 'cyan', 'maroon',
              'gold', 'lime']

    fig, ax = plt.subplots()
    ax.set_ylabel(y_label)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis='x', labelrotation=90)
    bplot = ax.boxplot(values, patch_artist=True)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.margins(0.1)
    plt.subplots_adjust(bottom=0.45)
    if is_best_data:
        plt.savefig(f"{graphs_dir}{type_data}_{type_of_dataset}_best.svg")
    else:
        plt.savefig(f"{graphs_dir}{type_data}_{type_of_dataset}.svg")
    

def generate_bar(models, values, type_of_data):
    colors = ['blue', 'green', 'red', 'purple', 'brown',
              'pink', 'gray', 'olive', 'cyan', 'maroon',
              'gold', 'lime']

    plt.figure()
    plt.bar(models, values, color = colors)
    if type_of_data == "not_json":
        plt.ylabel("Nenavráceno jako JSON")
    elif type_of_data == "precision":
        plt.ylabel("Precision")
    else:
        plt.ylabel("Recall")
    plt.xticks(rotation=90)
    plt.margins(0.1)
    plt.subplots_adjust(bottom=0.45)
    if is_best_data:
        plt.savefig(f"{graphs_dir}{type_of_data}_{type_of_dataset}_best.svg")
    else:
        plt.savefig(f"{graphs_dir}{type_of_data}_{type_of_dataset}.svg")

def generate_graph(type_of_data):
    tick_labels = []
    values = []
    y_label = ""

    if type_of_data == "correctness":
        for key in correctness_dict:
            tick_labels += [key]
            values += [correctness_dict[key]]
        y_label = "Správnost výsledku"
    elif type_of_data == "correct_data":
        for key in correct_data_count_dict:
            tick_labels += [key]
            values += [correct_data_count_dict[key]]
        y_label = "Počty správných dat"
    elif type_of_data == "incorrect_data":
        for key in incorrect_data_count_dict:
            tick_labels += [key]
            values += [incorrect_data_count_dict[key]]
        y_label = "Poměr špatnných dat"
    elif type_of_data == "not_found":
        for key in not_finded_main_count_key_dict:
            tick_labels += [key]
            values += [not_finded_main_count_key_dict[key]]
        y_label = "Poměr nenalezených dat"
    elif type_of_data == "goods_not_found":
        for key in goods_not_finded_count_dict:
            tick_labels += [key]
            values += [goods_not_finded_count_dict[key]]
        y_label = "Poměr nenalezených zboží"
    elif type_of_data == "time_of_run":
        for key in time_run_dict:
            tick_labels += [key]
            values += [time_run_dict[key]]
        y_label = "Délka běhu [s]"
    else:
        print("Not found type of data.")
        return
    
    generate_boxplot(tick_labels, values, y_label, type_of_data)

def generate_bar_graph_from_data(dict_data, type_of_data):
    names = []
    values = []

    for key in dict_data:
        names += [key]
        values += [float(dict_data[key]) / 103]
    
    generate_bar(names, values, type_of_data)


def load_all_data():
    for file in os.listdir(output_dir):
        model = file.split("_")[0]
        if model == "llava":
            model = "llava-7b"

        if not add_to_graph[model]:
            continue
        path_to_data = output_dir + file

        correctness_array = []
        correct_data_count_array = []
        incorrect_data_count_array = []
        not_finded_main_count_key_array = []
        goods_not_finded_count_array = []
        time_run_array = []
        precision_sum = 0.0
        recall_sum = 0.0

        with open(path_to_data, "r") as file:
            lines = file.readlines()

            for line in lines:
                array_of_values = line.replace("\n", "").split(";")

                correct_data = int(array_of_values[1])
                incorect_data = int(array_of_values[2])
                not_finded = int(array_of_values[3])
                if type_of_dataset == "ticket":
                    goods_not_finded = int(array_of_values[4])
                    count_of_all_data = get_count_of_all_data(correct_data, incorect_data, not_finded, goods_not_finded)
                else:
                    count_of_all_data = get_count_of_all_data(correct_data, incorect_data, not_finded, 0)
                
                correctness_array += [float(array_of_values[0])]
                correct_data_count_array += [correct_data]
                incorrect_data_count_array += [incorect_data / count_of_all_data]
                not_finded_main_count_key_array += [not_finded / count_of_all_data]

                if (correct_data + incorect_data) != 0:
                    precision_sum += (correct_data / (correct_data + incorect_data))
                if (correct_data + not_finded) != 0:
                    recall_sum += (correct_data / (correct_data + not_finded))

                if type_of_dataset == "ticket":
                    goods_not_finded_count_array += [(goods_not_finded * 3) / count_of_all_data]
                    time_run_array += [float(array_of_values[5])]
                else:
                    time_run_array += [float(array_of_values[4])]

                if type_of_dataset == "ticket":
                    if correct_data == 0 and incorect_data == 0 and not_finded > 0 and goods_not_finded == 0:
                        if model in not_found_json_dict:
                            not_found_json_dict[model] =  not_found_json_dict[model] + 1
                        else:
                            not_found_json_dict[model] = 1
                else:
                    if correct_data == 0 and incorect_data == 0 and not_finded > 0 and array_of_values[5] == "{}" and array_of_values[6] == "[]":
                        if model in not_found_json_dict:
                            not_found_json_dict[model] =  not_found_json_dict[model] + 1
                        else:
                            not_found_json_dict[model] = 1
                        
        
        correctness_dict[model] = correctness_array
        correct_data_count_dict[model] = correct_data_count_array
        incorrect_data_count_dict[model] = incorrect_data_count_array
        not_finded_main_count_key_dict[model] = not_finded_main_count_key_array
        goods_not_finded_count_dict[model] = goods_not_finded_count_array
        time_run_dict[model] = time_run_array
        precision_sum_dict[model] = precision_sum
        recall_sum_dict[model] = recall_sum
        

if __name__ == "__main__":
    load_all_data()

    if type_of_dataset == "ticket":
        generate_graph("correctness")
        generate_graph("incorrect_data")
        generate_graph("not_found")
        generate_graph("goods_not_found")
    generate_graph("time_of_run")

    if type_of_dataset == "objects":
        generate_bar_graph_from_data(precision_sum_dict, "precision")
        generate_bar_graph_from_data(recall_sum_dict, "recall")
    generate_bar_graph_from_data(not_found_json_dict, "not_json")