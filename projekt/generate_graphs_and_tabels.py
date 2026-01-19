import os
import matplotlib.pyplot as plt
import numpy as np

type_of_dataset = "ticket"
colors = ['blue', 'green', 'red', 'purple', 'brown',
              'pink', 'gray', 'olive', 'cyan', 'maroon',
              'gold', 'lime']

count_of_test_data_ocr = 103
count_of_test_data_objects = 100

add_to_graph = {
    "bakllava" : True,
    "easyocr": True,
    "gemini-2.0-flash-lite": True,
    "gemini-2.0-flash" : True,
    "gemini-2.5-flash-lite": True,
    "gemini-2.5-flash" : True,
    "gemini-2.5-pro" : True,
    "gemini-3-flash-preview" : True,
    "gemini-3-pro-preview" : True,
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
    "gpt-5.1": True,
    "gpt-5.2" : True,
    "granite3.2-vision" : True,
    "knoopx-mobile-vlm-3b-fp16" : True,
    "llava-7b" : True,
    "llava-13b" : True,
    "llava-34b" : True,
    "minicpm-v": True,
    "mistral-small3.1" : True,
    "mistral-small3.2-24b" : True,
    "tesseract-5.3.0" : True,
    "yolo11n" : True,
    "yolo11s" : True,
    "yolo11m" : True,
    "yolo11l" : True,
    "yolo11x" : True
    }

load_cpu_gpu_data = False
is_cpu_gpu_data_test = False
is_best_data = False
is_pattern_data = True

graphs_dir = "./graphs/"
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)

graphs_dir_patterns = "./graphs_patterns/"
if not os.path.exists(graphs_dir_patterns):
    os.makedirs(graphs_dir_patterns)

graphs_dir_patterns_objects = "./graphs_patterns_objects/"
if not os.path.exists(graphs_dir_patterns_objects):
    os.makedirs(graphs_dir_patterns_objects)

tables_dir = "./tables/"
if not os.path.exists(tables_dir):
    os.makedirs(tables_dir)

correctness_dict = {}
correct_data_count_dict = {}
incorrect_data_count_dict = {}
not_finded_main_count_key_dict = {}
goods_not_finded_count_dict = {}
time_run_dict = {}
not_found_json_dict = {}
not_found_json_object_dict = {}

correctness_tmp_dict = {}
correct_data_count_tmp_dict = {}
incorrect_data_count_tmp_dict = {}
not_finded_main_count_key_tmp_dict = {}
goods_not_finded_count_tmp_dict = {}
not_found_json_object_tmp_dict = {}

time_run_tmp_dict = {}
not_found_json_tmp_dict = {}

map_tmp_dict = {}
map_50_tmp_dict = {}
map_75_tmp_dict = {}
map_large_tmp_dict = {}
mar_100_tmp_dict = {}
mar_large_tmp_dict = {}

map_dict = {}
map_50_dict = {}
map_75_dict = {}
map_large_dict = {}
mar_100_dict = {}
mar_large_dict = {}

precision_sum_dict = {}
recall_sum_dict = {}

cpu_gpu_data = {}
cpu_gpu_data_time_diffs = {}

time_of_run_dict_tmp = {}
model_string_for_pattern = ""
count_of_data = 0

def set_output_dir():
    global output_dir
    global count_of_data

    if is_pattern_data:
        if type_of_dataset == "ticket":
            count_of_data = count_of_test_data_ocr
            output_dir = "./output_pattern_test/"
        else:
            count_of_data = count_of_test_data_objects
            output_dir = "./output_pattern_test_objects/"
    else:
        if load_cpu_gpu_data:
            if is_cpu_gpu_data_test:
                output_dir = "./test_measurement/"
            else:
                output_dir = "./train_measurement/"
        else:
            if type_of_dataset == "ticket":
                count_of_data = count_of_test_data_ocr
                output_dir = "./output/"
            else:
                count_of_test_data_objects
                output_dir = "./output_objects/"

def make_initial_structures():
    global model_string_for_pattern
    global count_of_data

    correctness_dict.clear()
    correct_data_count_dict.clear()
    incorrect_data_count_dict.clear()
    not_finded_main_count_key_dict.clear()
    goods_not_finded_count_dict.clear()
    time_run_dict.clear()
    not_found_json_dict.clear()
    not_found_json_object_dict.clear()

    correctness_tmp_dict.clear()
    correct_data_count_tmp_dict.clear()
    incorrect_data_count_tmp_dict.clear()
    not_finded_main_count_key_tmp_dict.clear()
    goods_not_finded_count_tmp_dict.clear()
    not_found_json_object_tmp_dict.clear()

    time_run_tmp_dict.clear()
    not_found_json_tmp_dict.clear()

    map_tmp_dict.clear()
    map_50_tmp_dict.clear()
    map_75_tmp_dict.clear()
    map_large_tmp_dict.clear()
    mar_100_tmp_dict.clear()
    mar_large_tmp_dict.clear()

    map_dict.clear()
    map_50_dict.clear()
    map_75_dict.clear()
    map_large_dict.clear()
    mar_100_dict.clear()
    mar_large_dict.clear()

    cpu_gpu_data.clear()
    cpu_gpu_data_time_diffs.clear()

    time_of_run_dict_tmp.clear()
    model_string_for_pattern = ""
    count_of_data = 0

def get_count_of_all_data(correct_data, incorect_data, not_finded, goods_not_finded):
    count_of_all_data = correct_data + incorect_data + not_finded + 3 * goods_not_finded
    return count_of_all_data

def count_number_of_types_jsons(output_array):
    count_of_zeros = 0
    count_of_ones = 0
    count_of_two = 0

    for value in output_array:
        if value == 0:
            count_of_zeros += 1
        elif value == 1:
            count_of_ones += 1
        elif value == 2:
            count_of_two += 1
        else:
            print("Incorrect json value.")
    
    return {
        "Korektní data" : count_of_zeros,
        "Objekty nenalezeny" : count_of_ones,
        "Špatná data" : count_of_two
        }

def generate_boxplot(tick_labels, values, y_label, type_data, model_name):
    fig, ax = plt.subplots()
    ax.set_ylabel(y_label)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis='x', labelrotation=90)
    bplot = ax.boxplot(values, patch_artist=True)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.margins(0.1)
    plt.subplots_adjust(bottom=0.45)

    if is_pattern_data:
        if type_of_dataset == "ticket":
            if is_best_data:
                plt.savefig(os.path.join(graphs_dir_patterns, f"{model_name}_{type_data}_best.svg"))
            else:
                plt.savefig(os.path.join(graphs_dir_patterns, f"{model_name}_{type_data}.svg"))
        else:
            if is_best_data:
                plt.savefig(os.path.join(graphs_dir_patterns_objects, f"{model_name}_{type_data}_best.svg"))
            else:
                plt.savefig(os.path.join(graphs_dir_patterns_objects, f"{model_name}_{type_data}.svg"))
    else:
        if is_best_data:
            plt.savefig(f"{graphs_dir}{type_data}_{type_of_dataset}_best.svg")
        elif load_cpu_gpu_data and not is_cpu_gpu_data_test:
            plt.savefig(f"{graphs_dir}{type_data}_{type_of_dataset}_train.svg")
        elif load_cpu_gpu_data and is_cpu_gpu_data_test:
            plt.savefig(f"{graphs_dir}{type_data}_{type_of_dataset}_test.svg")
        else:
            plt.savefig(f"{graphs_dir}{type_data}_{type_of_dataset}.svg")

def generate_bar(models, values, type_of_data):
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

    if is_pattern_data:
        if type_of_dataset == "ticket":
            graph_path = os.path.join(graphs_dir_patterns, f"{model_string_for_pattern}_{type_of_data}")
        else:
            graph_path = os.path.join(graphs_dir_patterns_objects, f"{model_string_for_pattern}_{type_of_data}")
    else:
        if load_cpu_gpu_data and not is_cpu_gpu_data_test:
            graph_path = f"{graphs_dir}{type_of_data}_{type_of_dataset}_train"
        elif load_cpu_gpu_data and is_cpu_gpu_data_test:
            graph_path = f"{graphs_dir}{type_of_data}_{type_of_dataset}_test"
        else:
            graph_path = f"{graphs_dir}{type_of_data}_{type_of_dataset}"

    if is_best_data:
        graph_path = f"{graph_path}_best.svg"
    else:
        graph_path = f"{graph_path}.svg"
    
    if os.path.exists(graph_path):
        os.remove(graph_path)
    plt.savefig(graph_path)

def generate_grouped_bar_objects(dict_data, model, type_of_data):
    categories = list(dict_data.keys())
    labels = []
    
    for category in dict_data:
        for label in dict_data[category]:
            if label not in labels:
                labels.append(label)

    values_dict = {label: [] for label in labels}
    for category in categories:
        for label in labels:
            values_dict[label].append(dict_data[category].get(label, 0))

    x = np.arange(len(categories))
    
    n_groups = len(labels)
    total_width = 0.8
    width = total_width / n_groups

    fig, ax = plt.subplots()
    parts_of_graphs = []

    for index, label in enumerate(labels):
        offset = (index - (n_groups - 1) / 2) * width
        values = values_dict[label]

        part = ax.bar(x + offset, values, width, label=label, color=colors[index % len(colors)])
        parts_of_graphs.append(part)

    if type_of_data == "count_of_data":
        ax.set_ylabel("Počty dat")
    else:
        ax.set_ylabel(type_of_data)

    ax.set_xlabel('Vstupní textové zadání')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
    ax.legend()

    for part in parts_of_graphs:
        ax.bar_label(part, padding=3)

    plt.tight_layout()
    if is_pattern_data:
        graph_path = os.path.join(graphs_dir_patterns_objects, f"{model}_{type_of_data}")
    else:
        return
    
    if is_best_data:
        graph_path = f"{graph_path}_best.svg"
    else:
        graph_path = f"{graph_path}.svg"
    
    if os.path.exists(graph_path):
        os.remove(graph_path)
    plt.savefig(graph_path)

def generate_latex_table_and_save_to_file(type_of_data):
    if load_cpu_gpu_data and not is_cpu_gpu_data_test:
        output_file_path = f"{tables_dir}{type_of_data}_{type_of_dataset}_train"
    elif load_cpu_gpu_data and is_cpu_gpu_data_test:
        output_file_path = f"{tables_dir}{type_of_data}_{type_of_dataset}_test"
    else:
        output_file_path = f"{tables_dir}{type_of_data}_{type_of_dataset}"
    
    if is_best_data:
        output_file_path = f"{output_file_path}_best.txt"
    else:
        output_file_path = f"{output_file_path}.txt"

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    with open(output_file_path, "+a") as file:
        file.write("\\begin{table}[h!]\n")
        file.write("\\begin{tabular}{|l|l|l|l|l|}\n")
        file.write("\\hline\n")

        if type_of_data == "cpu_usage_main_thread":
            file.write("\\textbf{Model} & \\textbf{CPU [%]} & \\textbf{CPU MIN} & \\textbf{CPU MAX} & \\textbf{CPU Medián} \\\\ \hline\n")
            for model in cpu_gpu_data:
                cpu_values = cpu_gpu_data[model]["cpu_usage"]
                cpu_min = min(cpu_values)
                cpu_max = max(cpu_values)
                cpu_median = sorted(cpu_values)[len(cpu_values) // 2]
                cpu_avg = sum(cpu_values) / len(cpu_values)
                file.write(f"{model} & {cpu_avg:.2f} & {cpu_min:.2f} & {cpu_max:.2f} & {cpu_median:.2f} \\\\ \hline\n")
        elif type_of_data == "cpu_usage_peak":
            file.write("\\textbf{Model} & \\textbf{CPU [%]} & \\textbf{CPU MIN} & \\textbf{CPU MAX} & \\textbf{CPU Medián} \\\\ \hline\n")
            for model in cpu_gpu_data:
                cpu_values = cpu_gpu_data[model]["peak_cpu_percent"]
                cpu_min = min(cpu_values)
                cpu_max = max(cpu_values)
                cpu_median = sorted(cpu_values)[len(cpu_values) // 2]
                cpu_avg = sum(cpu_values) / len(cpu_values)
                file.write(f"{model} & {cpu_avg:.2f} & {cpu_min:.2f} & {cpu_max:.2f} & {cpu_median:.2f} \\\\ \hline\n")
        elif type_of_data == "ram_usage_peak":
            file.write("\\textbf{Model} & \\textbf{RAM} & \\textbf{RAM MIN} & \\textbf{RAM MAX} & \\textbf{RAM Medián} \\\\ \hline\n")
            for model in cpu_gpu_data:
                ram_values = cpu_gpu_data[model]["ram_usage"]
                ram_min = min(ram_values)
                ram_max = max(ram_values)
                ram_median = sorted(ram_values)[len(ram_values) // 2]
                ram_avg = sum(ram_values) / len(ram_values)
                file.write(f"{model} & {ram_avg:.2f} & {ram_min:.2f} & {ram_max:.2f} & {ram_median:.2f} \\\\ \hline\n")
        elif type_of_data == "gpu_usage":
            file.write("\\textbf{Model} & \\textbf{GPU [%]} & \\textbf{GPU MIN} & \\textbf{GPU MAX} & \\textbf{GPU Medián} \\\\ \hline\n")
            for model in cpu_gpu_data:
                gpu_values = cpu_gpu_data[model]["peak_gpu_utilization"]
                gpu_min = min(gpu_values)
                gpu_max = max(gpu_values)
                gpu_median = sorted(gpu_values)[len(gpu_values) // 2]
                gpu_avg = sum(gpu_values) / len(gpu_values)
                file.write(f"{model} & {gpu_avg:.2f} & {gpu_min:.2f} & {gpu_max:.2f} & {gpu_median:.2f} \\\\ \hline\n")
        elif type_of_data == "vram_usage":
            file.write("\\textbf{Model} & \\textbf{VRAM} & \\textbf{VRAM MIN} & \\textbf{VRAM MAX} & \\textbf{VRAM Medián} \\\\ \hline\n")
            for model in cpu_gpu_data:
                vram_values = cpu_gpu_data[model]["total_vram_mb"]
                vram_min = min(vram_values)
                vram_max = max(vram_values)
                vram_median = sorted(vram_values)[len(vram_values) // 2]
                vram_avg = sum(vram_values) / len(vram_values)
                file.write(f"{model} & {vram_avg:.2f} & {vram_min:.2f} & {vram_max:.2f} & {vram_median:.2f} \\\\ \hline\n")
        elif type_of_data == "time_of_run":
            file.write("\\textbf{Model} & \\textbf{Čas [%]} & \\textbf{Čas MIN} & \\textbf{Čas MAX} & \\textbf{Čas Medián} \\\\ \hline\n")
            for model in time_of_run_dict_tmp:
                time_values = time_of_run_dict_tmp[model]
                time_min = min(time_values)
                time_max = max(time_values)
                time_median = sorted(time_values)[len(time_values) // 2]
                time_avg = sum(time_values) / len(time_values)
                file.write(f"{model} & {time_avg:.2f} & {time_min:.2f} & {time_max:.2f} & {time_median:.2f} \\\\ \hline\n")
        else:
            print("Not found type of data.")
            return
        
        file.write("\\end{tabular}\n")
        file.write("\\centering\n")

        if type_of_data == "cpu_usage_main_thread":
            if load_cpu_gpu_data and not is_cpu_gpu_data_test:
                if is_best_data:
                    file.write("\\caption{Využití procesoru z hlavního vlákna (trénování modelu, nejlepší výsledky)}\n")
                    file.write("\\label{tab: cpu_usage_main_thread_train_best}\n")
                else:
                    file.write("\\caption{Využití procesoru z hlavního vlákna (trénování modelu)}\n")
                    file.write("\\label{tab: cpu_usage_main_thread_train}\n")
            elif load_cpu_gpu_data and is_cpu_gpu_data_test:
                if is_best_data:
                    file.write("\\caption{Využití procesoru z hlavního vlákna (testování modelu, nejlepší výsledky)}\n")
                    file.write("\\label{tab: cpu_usage_main_thread_test_best}\n")
                else:
                    file.write("\\caption{Využití procesoru z hlavního vlákna (testování modelu)}\n")
                    file.write("\\label{tab: cpu_usage_main_thread_test}\n")
            else:
                if is_best_data:
                    file.write("\\caption{Využití procesoru z hlavního vlákna (nejlepší výsledky)")
                    file.write("\\label{tab: cpu_usage_main_thread_best}\n")
                else:
                    file.write("\\caption{Využití procesoru z hlavního vlákna")
                    file.write("\\label{tab: cpu_usage_main_thread}\n")
        elif type_of_data == "cpu_usage_peak":
            if load_cpu_gpu_data and not is_cpu_gpu_data_test:
                if is_best_data:
                    file.write("\\caption{Využití procesoru z hlavního vlákna i vedlejších vláken (trénování modelu, nejlepší výsledky)}\n")
                    file.write("\\label{tab: cpu_usage_peak_train_best}\n")
                else:
                    file.write("\\caption{Využití procesoru z hlavního vlákna i vedlejších vláken (trénování modelu)}\n")
                    file.write("\\label{tab: cpu_usage_peak_train}\n")
            elif load_cpu_gpu_data and is_cpu_gpu_data_test:
                if is_best_data:
                    file.write("\\caption{Využití procesoru z hlavního vlákna i vedlejších vláken (testování modelu, nejlepší výsledky)}\n")
                    file.write("\\label{tab: cpu_usage_peak_test_best}\n")
                else:
                    file.write("\\caption{Využití procesoru z hlavního vlákna i vedlejších vláken (testování modelu)}\n")
                    file.write("\\label{tab: cpu_usage_peak_test}\n")
            else:
                if is_best_data:
                    file.write("\\caption{Využití procesoru z hlavního vlákna i vedlejších vláken (nejlepší výsledky)")
                    file.write("\\label{tab: cpu_usage_peak_best}\n")
                else:
                    file.write("\\caption{Využití procesoru z hlavního vlákna i vedlejších vláken")
                    file.write("\\label{tab: cpu_usage_peak}\n")
        elif type_of_data == "ram_usage_peak":
            if load_cpu_gpu_data and not is_cpu_gpu_data_test:
                if is_best_data:
                    file.write("\\caption{Využití paměti RAM (trénování modelu, nejlepší výsledky)}\n")
                    file.write("\\label{tab: ram_usage_peak_train_best}\n")
                else:
                    file.write("\\caption{Využití paměti RAM (trénování modelu)}\n")
                    file.write("\\label{tab: ram_usage_peak_train}\n")
            elif load_cpu_gpu_data and is_cpu_gpu_data_test:
                if is_best_data:
                    file.write("\\caption{Využití paměti RAM (testování modelu, nejlepší výsledky)}\n")
                    file.write("\\label{tab: ram_usage_peak_test_best}\n")
                else:
                    file.write("\\caption{Využití paměti RAM (testování modelu)}\n")
                    file.write("\\label{tab: ram_usage_peak_test}\n")
            else:
                if is_best_data:
                    file.write("\\caption{Využití paměti RAM (nejlepší výsledky)")
                    file.write("\\label{tab: ram_usage_peak_best}\n")
                else:
                    file.write("\\caption{Využití paměti RAM")
                    file.write("\\label{tab: ram_usage_peak}\n")
        elif type_of_data == "gpu_usage":
            if load_cpu_gpu_data and not is_cpu_gpu_data_test:
                if is_best_data:
                    file.write("\\caption{Využití grafické karty (trénování modelu, nejlepší výsledky)}\n")
                    file.write("\\label{tab: gpu_usage_train_best}\n")
                else:
                    file.write("\\caption{Využití grafické karty (trénování modelu)}\n")
                    file.write("\\label{tab: gpu_usage_train}\n")
            elif load_cpu_gpu_data and is_cpu_gpu_data_test:
                if is_best_data:
                    file.write("\\caption{Využití grafické karty (testování modelu, nejlepší výsledky)}\n")
                    file.write("\\label{tab: gpu_usage_test_best}\n")
                else:
                    file.write("\\caption{Využití grafické karty (testování modelu)}\n")
                    file.write("\\label{tab: gpu_usage_test}\n")
            else:
                if is_best_data:
                    file.write("\\caption{Využití grafické karty (nejlepší výsledky)")
                    file.write("\\label{tab: gpu_usage_best}\n")
                else:
                    file.write("\\caption{Využití grafické karty")
                    file.write("\\label{tab: gpu_usage}\n")
        elif type_of_data == "vram_usage":
            if load_cpu_gpu_data and not is_cpu_gpu_data_test:
                if is_best_data:
                    file.write("\\caption{Využití paměti VRAM (trénování modelu, nejlepší výsledky)}\n")
                    file.write("\\label{tab: vram_usage_train_best}\n")
                else:
                    file.write("\\caption{Využití paměti VRAM (trénování modelu)}\n")
                    file.write("\\label{tab: vram_usage_train}\n")
            elif load_cpu_gpu_data and is_cpu_gpu_data_test:
                if is_best_data:
                    file.write("\\caption{Využití paměti VRAM (testování modelu, nejlepší výsledky)}\n")
                    file.write("\\label{tab: vram_usage_test_best}\n")
                else:
                    file.write("\\caption{Využití paměti VRAM (testování modelu)}\n")
                    file.write("\\label{tab: vram_usage_test}\n")
            else:
                if is_best_data:
                    file.write("\\caption{Využití paměti VRAM (nejlepší výsledky)")
                    file.write("\\label{tab: vram_usage_best}\n")
                else:
                    file.write("\\caption{Využití paměti VRAM")
                    file.write("\\label{tab: vram_usage}\n")
        elif type_of_data == "time_of_run":
            if load_cpu_gpu_data and not is_cpu_gpu_data_test:
                if is_best_data:
                    file.write("\\caption{Délka běhu [s] (trénování modelu, nejlepší výsledky)}\n")
                    file.write("\\label{tab: time_of_run_train_best}\n")
                else:
                    file.write("\\caption{Délka běhu [s] (trénování modelu)}\n")
                    file.write("\\label{tab: time_of_run_train}\n")
            elif load_cpu_gpu_data and is_cpu_gpu_data_test:
                if is_best_data:
                    file.write("\\caption{Délka běhu [s] (testování modelu, nejlepší výsledky)}\n")
                    file.write("\\label{tab: time_of_run_test_best}\n")
                else:
                    file.write("\\caption{Délka běhu [s] (testování modelu)}\n")
                    file.write("\\label{tab: time_of_run_test}\n")
            else:
                if is_best_data:
                    file.write("\\caption{Délka běhu [s] (nejlepší výsledky)")
                    file.write("\\label{tab: time_of_run_best}\n")
                else:
                    file.write("\\caption{Délka běhu [s]")
                    file.write("\\label{tab: time_of_run}\n")

        file.write("\\end{table}\n")

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
    elif type_of_data == "cpu_usage_main_thread":
        for key in cpu_gpu_data:
            tick_labels += [key]
            values += [cpu_gpu_data[key]["cpu_usage"]]
        y_label = "Využití CPU (hlavní vlákno)"
    elif type_of_data == "cpu_usage_peak":
        for key in cpu_gpu_data:
            tick_labels += [key]
            values += [cpu_gpu_data[key]["peak_cpu_percent"]]
        y_label = "Využití CPU (nejvyšší zatížení)"
    elif type_of_data == "ram_usage_peak":
        for key in cpu_gpu_data:
            tick_labels += [key]
            values += [cpu_gpu_data[key]["ram_usage"]]
        y_label = "Využití RAM (nejvyšší spotřeba)"
    elif type_of_data == "gpu_usage":
        for key in cpu_gpu_data:
            tick_labels += [key]
            values += [cpu_gpu_data[key]["peak_gpu_utilization"]]
        y_label = "Využití GPU"
    elif type_of_data == "vram_usage":
        for key in cpu_gpu_data:
            tick_labels += [key]
            values += [cpu_gpu_data[key]["total_vram_mb"]]
        y_label = "Využití VRAM"
    elif type_of_data == "time_of_run_cpu_gpu":
        for key in cpu_gpu_data_time_diffs:
            tick_labels += [key]
            values += [cpu_gpu_data_time_diffs[key]]
        y_label = "Délka běhu [s]"
    else:
        print("Not found type of data.")
        return
    
    if is_pattern_data:
        generate_boxplot(tick_labels, values, y_label, type_of_data, model_string_for_pattern)
    else:
        generate_boxplot(tick_labels, values, y_label, type_of_data, "")

def generate_bar_graph_from_data(dict_data, type_of_data):
    names = []
    values = []

    for key in dict_data:
        names += [key]
        values += [float(dict_data[key]) / count_of_data]
    
    generate_bar(names, values, type_of_data)

def load_output_of_models_base(file_path, pattern_directory):
    if pattern_directory == "":
        path_to_data = os.path.join(output_dir, file_path)
    else:
        path_to_data = os.path.join(output_dir, pattern_directory, file_path)

    correctness_array = []
    correct_data_count_array = []
    incorrect_data_count_array = []
    not_finded_main_count_key_array = []
    goods_not_finded_count_array = []
    time_run_array = []
    not_found_json = 0

    with open(path_to_data, "r") as file:
        lines = file.readlines()

        for line in lines:
            array_of_values = line.replace("\n", "").split(";")

            correct_data = int(array_of_values[1])
            incorect_data = int(array_of_values[2])
            not_finded = int(array_of_values[3])
            goods_not_finded = int(array_of_values[4])
            count_of_all_data = get_count_of_all_data(correct_data, incorect_data, not_finded, goods_not_finded)
                
            correctness_array += [float(array_of_values[0])]
            correct_data_count_array += [correct_data]
            incorrect_data_count_array += [incorect_data / count_of_all_data]
            not_finded_main_count_key_array += [not_finded / count_of_all_data]

            goods_not_finded_count_array += [(goods_not_finded * 3) / count_of_all_data]
            time_run_array += [float(array_of_values[5])]

            if correct_data == 0 and incorect_data == 0 and not_finded > 0 and goods_not_finded == 0:
                not_found_json =  not_found_json + 1
    
    return (correctness_array, correct_data_count_array,
            incorrect_data_count_array, not_finded_main_count_key_array,
            goods_not_finded_count_array, time_run_array, not_found_json)

def add_to_correctness_dict_pattern(model_name, pattern_name, value):
    if model_name not in correctness_tmp_dict:
        correctness_tmp_dict[model_name] = {pattern_name: value}
    else:
        correctness_tmp_dict[model_name][pattern_name] = value

def add_to_incorrect_data_dict_pattern(model_name, pattern_name, value):
    if model_name not in incorrect_data_count_tmp_dict:
        incorrect_data_count_tmp_dict[model_name] = {pattern_name: value}
    else:
        incorrect_data_count_tmp_dict[model_name][pattern_name] = value

def add_to_not_founded_data_dict_pattern(model_name, pattern_name, value):
    if model_name not in not_finded_main_count_key_tmp_dict:
        not_finded_main_count_key_tmp_dict[model_name] = {pattern_name: value}
    else:
        not_finded_main_count_key_tmp_dict[model_name][pattern_name] = value

def add_to_goods_not_founded_data_dict_pattern(model_name, pattern_name, value):
    if model_name not in goods_not_finded_count_tmp_dict:
        goods_not_finded_count_tmp_dict[model_name] = {pattern_name: value}
    else:
        goods_not_finded_count_tmp_dict[model_name][pattern_name] = value

def add_to_time_run_dict_pattern(model_name, pattern_name, value):
    if model_name not in time_run_tmp_dict:
        time_run_tmp_dict[model_name] = {pattern_name: value}
    else:
        time_run_tmp_dict[model_name][pattern_name] = value

def add_not_found_json_dict_pattern(model_name, pattern_name, value):
    if model_name not in not_found_json_tmp_dict:
        not_found_json_tmp_dict[model_name] = {pattern_name: value}
    else:
        not_found_json_tmp_dict[model_name][pattern_name] = value

def load_output_of_models(file_path, model_name):
    data = load_output_of_models_base(file_path, "")
                        
    correctness_dict[model_name] = data[0]
    correct_data_count_dict[model_name] = data[1]
    incorrect_data_count_dict[model_name] = data[2]
    not_finded_main_count_key_dict[model_name] = data[3]
    goods_not_finded_count_dict[model_name] = data[4]
    time_run_dict[model_name] = data[5]
    not_found_json_dict[model_name] = data[6]

def load_output_of_models_objects_base(file_path, pattern_directory):
    if pattern_directory == "":
        path_to_data = os.path.join(output_dir, file_path)
    else:
        path_to_data = os.path.join(output_dir, pattern_directory, file_path)

    map_sum = 0.0
    map_50_sum = 0.0
    map_75_sum = 0.0
    map_large_sum = 0.0
    mar_100_sum = 0.0
    mar_large_sum = 0.0
    
    number_of_entries = 0

    with open(path_to_data, "r") as file:
        lines = file.readlines()

        for line in lines:
            array_of_values = line.replace("\n", "").split(";")

            map_sum += float(array_of_values[0])
            map_50_sum += float(array_of_values[1])
            map_75_sum += float(array_of_values[2])
            map_large_sum += float(array_of_values[3])
            mar_100_sum += float(array_of_values[4])
            mar_large_sum += float(array_of_values[5])

            number_of_entries += 1
    
    return (map_sum, map_50_sum, map_75_sum, map_large_sum,
            mar_100_sum, mar_large_sum, number_of_entries)

def add_to_map_dict(model_name, iou, value):
    if model_name not in map_dict:
        map_dict[model_name] = {iou: value}
    else:
        map_dict[model_name][iou] = value

def add_to_map_dict_transform(dictionary, pattern, iou, value):
    if pattern not in dictionary:
        dictionary[pattern] = {iou: value}
    else:
        dictionary[pattern][iou] = value
    
    return dictionary

def add_to_map_50_dict(model_name, iou, value):
    if model_name not in map_50_dict:
        map_50_dict[model_name] = {iou: value}
    else:
        map_50_dict[model_name][iou] = value

def add_to_map_50_dict_transform(dictionary, pattern, iou, value):
    if pattern not in dictionary:
        dictionary[pattern] = {iou: value}
    else:
        dictionary[pattern][iou] = value
    
    return dictionary

def add_to_map_75_dict(model_name, iou, value):
    if model_name not in map_75_dict:
        map_75_dict[model_name] = {iou: value}
    else:
        map_75_dict[model_name][iou] = value

def add_to_map_75_dict_transform(dictionary, pattern, iou, value):
    if pattern not in dictionary:
        dictionary[pattern] = {iou: value}
    else:
        dictionary[pattern][iou] = value

    return dictionary

def add_to_map_large_dict(model_name, iou, value):
    if model_name not in map_large_dict:
        map_large_dict[model_name] = {iou: value}
    else:
        map_large_dict[model_name][iou] = value

def add_to_map_large_dict_transform(dictionary, pattern, iou, value):
    if pattern not in dictionary:
        dictionary[pattern] = {iou: value}
    else:
        dictionary[pattern][iou] = value
    
    return dictionary

def add_to_mar_100_dict(model_name, iou, value):
    if model_name not in mar_100_dict:
        mar_100_dict[model_name] = {iou: value}
    else:
        mar_100_dict[model_name][iou] = value

def add_to_mar_100_dict_transform(dictionary, pattern, iou, value):
    if pattern not in dictionary:
        dictionary[pattern] = {iou: value}
    else:
        dictionary[pattern][iou] = value
    
    return dictionary

def add_to_mar_large_dict(model_name, iou, value):
    if model_name not in mar_large_dict:
        mar_large_dict[model_name] = {iou: value}
    else:
        mar_large_dict[model_name][iou] = value

def add_to_mar_large_dict_transform(dictionary, pattern, iou, value):
    if pattern not in dictionary:
        dictionary[pattern] = {iou: value}
    else:
        dictionary[pattern][iou] = value
    
    return dictionary

def load_output_of_models_objects(file_path, model_name, iou):
    data = load_output_of_models_objects_base(file_path, "")
    
    add_to_map_dict(model_name, iou, data[0] / data[6])
    add_to_map_50_dict(model_name, iou, data[1] / data[6])
    add_to_map_75_dict(model_name, iou, data[2] / data[6])
    add_to_map_large_dict(model_name, iou, data[3] / data[6])
    add_to_mar_100_dict(model_name, iou, data[4] / data[6])
    add_to_mar_large_dict(model_name, iou, data[5] / data[6])

def load_output_of_models_objects_main_base(file_path, pattern_directory):
    if pattern_directory == "":
        path_to_data = os.path.join(output_dir, file_path)
    else:
        path_to_data = os.path.join(output_dir, pattern_directory, file_path)

    array_of_time_of_run = []
    json_loading_array = []

    with open(path_to_data, "r") as file:
        lines = file.readlines()

        for line in lines:
            array_of_values = line.replace("\n", "").split(";")
            array_of_time_of_run += [float(array_of_values[0])]
            json_loading_array += [float(array_of_values[1])]

    
    return array_of_time_of_run, json_loading_array

def add_to_time_of_run_dict(model_name, value):
    time_run_dict[model_name] = value

def add_to_time_of_run_dict_transform(dictionary, pattern, value):
    dictionary[pattern] = value

    return dictionary

def add_to_not_found_json_object_dict(model_name, value):
    not_found_json_object_dict[model_name] = value

def add_to_not_found_json_tmp_dict_transform(dictionary, pattern, value):
    dictionary[pattern] = value

    return dictionary

def load_output_of_models_objects_main(file_path, model_name):
    data = load_output_of_models_objects_main_base(file_path, "")

    add_to_time_of_run_dict(model_name, data[0])
    add_to_not_found_json_object_dict(model_name, data[1])

def load_cpu_gpu_data_of_models(file_path, model_name):
    path_to_data = output_dir + file_path

    with open(path_to_data, "r") as file:
        lines = file.readlines()

        for line in lines:
            array_of_values = line.replace("\n", "").split(";")
            
            if model_name not in cpu_gpu_data:
                cpu_gpu_data[model_name] = {
                    "cpu_usage": [],
                    "peak_cpu_percent": [],
                    "ram_usage": [],
                    "peak_gpu_utilization": [],
                    "total_vram_mb": []
                }
                cpu_gpu_data_time_diffs[model_name] = []
            
            cpu_gpu_data[model_name]["cpu_usage"] += [float(array_of_values[0]) / 100]
            cpu_gpu_data[model_name]["peak_cpu_percent"] += [float(array_of_values[1]) / 100]
            cpu_gpu_data[model_name]["ram_usage"] += [float(array_of_values[2])]
            cpu_gpu_data[model_name]["peak_gpu_utilization"] += [float(array_of_values[3]) / 100]
            cpu_gpu_data[model_name]["total_vram_mb"] += [float(array_of_values[4])]

            if not is_cpu_gpu_data_test:
                cpu_gpu_data_time_diffs[model_name] += [float(array_of_values[5])]


def load_all_data():
    for file in os.listdir(output_dir):
        model = file.split("_")[0]
        if model == "llava":
            model = "llava-7b"

        if not add_to_graph[model]:
            continue
        
        if load_cpu_gpu_data:
            if file.split("_")[1].split(".")[0] == type_of_dataset:
                load_cpu_gpu_data_of_models(file, model)
        else:
            if type_of_dataset == "ticket":
                load_output_of_models(file, model)
            else:
                file_split = file.split("_")
                if len(file_split) == 3:
                    third = file_split[2].replace(".txt", "")
                    if third == "main":
                        load_output_of_models_objects_main(file, model)
                    else:
                        load_output_of_models_objects(file, model, third)

def load_output_of_models_pattern(file_path, model_name, pattern_name):
    data = load_output_of_models_base(file_path, pattern_name)

    add_to_correctness_dict_pattern(model_name, pattern_name, data[0])
    correct_data_count_tmp_dict[model_name] = {pattern_name: data[1]}
    add_to_incorrect_data_dict_pattern(model_name, pattern_name, data[2])
    add_to_not_founded_data_dict_pattern(model_name, pattern_name, data[3])
    add_to_goods_not_founded_data_dict_pattern(model_name, pattern_name, data[4])

    add_to_time_run_dict_pattern(model_name, pattern_name, data[5])
    add_not_found_json_dict_pattern(model_name, pattern_name, data[6])

def load_output_of_models_objects_main_pattern(file_path, model_name, pattern_name):
    time_of_run_dict_array, not_found_json_object_array = load_output_of_models_objects_main_base(file_path, pattern_name)
    
    add_to_time_run_dict_pattern(model_name, pattern_name, time_of_run_dict_array)
    add_not_found_json_dict_pattern(model_name, pattern_name, not_found_json_object_array)

def load_output_of_models_objects_pattern(file_path, model_name, pattern_name, iou):
    data = load_output_of_models_objects_base(file_path, pattern_name)

    if model_name not in map_tmp_dict:
        map_tmp_dict[model_name] = {pattern_name: {iou: data[0] / data[6]}}
    else:
        if pattern_name not in map_tmp_dict[model_name]:
            map_tmp_dict[model_name][pattern_name] = {iou: data[0] / data[6]}
        else:
            map_tmp_dict[model_name][pattern_name][iou] = data[0] / data[6]
    
    if model_name not in map_50_tmp_dict:
        map_50_tmp_dict[model_name] = {pattern_name: {iou: data[1] / data[6]}}
    else:
        if pattern_name not in map_50_tmp_dict[model_name]:
            map_50_tmp_dict[model_name][pattern_name] = {iou: data[1] / data[6]}
        else:
            map_50_tmp_dict[model_name][pattern_name][iou] = data[1] / data[6]

    if model_name not in map_75_tmp_dict:
        map_75_tmp_dict[model_name] = {pattern_name: {iou: data[2] / data[6]}}
    else:
        if pattern_name not in map_75_tmp_dict[model_name]:
            map_75_tmp_dict[model_name][pattern_name] = {iou: data[2] / data[6]}
        else:
            map_75_tmp_dict[model_name][pattern_name][iou] = data[2] / data[6]

    if model_name not in map_large_tmp_dict:
        map_large_tmp_dict[model_name] = {pattern_name: {iou: data[3] / data[6]}}
    else:
        if pattern_name not in map_large_tmp_dict[model_name]:
            map_large_tmp_dict[model_name][pattern_name] = {iou: data[3] / data[6]}
        else:
            map_large_tmp_dict[model_name][pattern_name][iou] = data[3] / data[6]

    if model_name not in mar_100_tmp_dict:
        mar_100_tmp_dict[model_name] = {pattern_name: {iou: data[4] / data[6]}}
    else:
        if pattern_name not in mar_100_tmp_dict[model_name]:
            mar_100_tmp_dict[model_name][pattern_name] = {iou: data[4] / data[6]}
        else:
            mar_100_tmp_dict[model_name][pattern_name][iou] = data[4] / data[6]

    if model_name not in mar_large_tmp_dict:
        mar_large_tmp_dict[model_name] = {pattern_name: {iou: data[5] / data[6]}}
    else:
        if pattern_name not in mar_large_tmp_dict[model_name]:
            mar_large_tmp_dict[model_name][pattern_name] = {iou: data[5] / data[6]}
        else:
            mar_large_tmp_dict[model_name][pattern_name][iou] = data[5] / data[6]

def load_all_data_pattern():
    for pattern in os.listdir(output_dir):
        path_to_pattern_dir = os.path.join(output_dir, pattern)
        for file in os.listdir(path_to_pattern_dir):
            model = file.split("_")[0]
            if model == "llava":
                model = "llava-7b"

            if not add_to_graph[model]:
                continue

            if type_of_dataset == "ticket":
                load_output_of_models_pattern(file, model, pattern)
            else:
                file_split = file.split("_")
                if len(file_split) == 3:
                    third = file_split[2].replace(".txt", "")
                    if third == "main":
                        load_output_of_models_objects_main_pattern(file, model, pattern)
                    else:
                        load_output_of_models_objects_pattern(file, model, pattern, third)

def transform_object_pattern_data_to_normal():
    map_array = []
    map_50_array = []
    map_75_array = []
    map_large_array = []
    mar_100_array = []
    mar_large_array = []

    for model in map_tmp_dict:
        map_dict = {}
        for pattern in map_tmp_dict[model]:
            for iou in map_tmp_dict[model][pattern]:
                add_to_map_dict_transform(map_dict, pattern, iou, map_tmp_dict[model][pattern][iou])
        map_array.append({model: map_dict})
    
    for model in map_50_tmp_dict:
        map_50_dict = {}
        for pattern in map_50_tmp_dict[model]:
            for iou in map_50_tmp_dict[model][pattern]:
                add_to_map_50_dict_transform(map_50_dict, pattern, iou, map_50_tmp_dict[model][pattern][iou])
        map_50_array.append({model: map_50_dict})
    
    for model in map_75_tmp_dict:
        map_75_dict = {}
        for pattern in map_75_tmp_dict[model]:
            for iou in map_75_tmp_dict[model][pattern]:
                add_to_map_75_dict_transform(map_75_dict, pattern, iou, map_75_tmp_dict[model][pattern][iou])
        map_75_array.append({model: map_75_dict})
    
    for model in map_large_tmp_dict:
        map_large_dict = {}
        for pattern in map_large_tmp_dict[model]:
            for iou in map_large_tmp_dict[model][pattern]:
                add_to_map_large_dict_transform(map_large_dict, pattern, iou, map_large_tmp_dict[model][pattern][iou])
        map_large_array.append({model: map_large_dict})
    
    for model in mar_100_tmp_dict:
        mar_100_dict = {}
        for pattern in mar_100_tmp_dict[model]:
            for iou in mar_100_tmp_dict[model][pattern]:
                add_to_mar_100_dict_transform(mar_100_dict, pattern, iou, mar_100_tmp_dict[model][pattern][iou])
        mar_100_array.append({model: mar_100_dict})
    
    for model in mar_large_tmp_dict:
        mar_large_dict = {}
        for pattern in mar_large_tmp_dict[model]:
            for iou in mar_large_tmp_dict[model][pattern]:
                add_to_mar_large_dict_transform(mar_large_dict, pattern, iou, mar_large_tmp_dict[model][pattern][iou])
        mar_large_array.append({model: mar_large_dict})
    
    return (map_array, map_50_array, map_75_array,
            map_large_array, mar_100_array, mar_large_array)

def transform_ocr_pattern_data_to_normal():
    correctness_array = []
    correct_data_count_array = []
    incorrect_data_count_array = []
    not_finded_main_count_key_array = []
    goods_not_finded_count_array = []

    for model in correctness_tmp_dict:
        correctness_dict = {}
        for pattern in correctness_tmp_dict[model]:
            correctness_dict[pattern] = correctness_tmp_dict[model][pattern]
        correctness_array.append({model: correctness_dict})
    
    for model in correct_data_count_tmp_dict:
        correct_data_count_dict = {}
        for pattern in correct_data_count_tmp_dict[model]:
            correct_data_count_dict[pattern] = correct_data_count_tmp_dict[model][pattern]
        correct_data_count_array.append({model: correct_data_count_dict})
    
    for model in incorrect_data_count_tmp_dict:
        incorrect_data_count_dict = {}
        for pattern in incorrect_data_count_tmp_dict[model]:
            incorrect_data_count_dict[pattern] = incorrect_data_count_tmp_dict[model][pattern]
        incorrect_data_count_array.append({model: incorrect_data_count_dict})
    
    for model in not_finded_main_count_key_tmp_dict:
        not_finded_main_count_key_dict = {}
        for pattern in not_finded_main_count_key_tmp_dict[model]:
            not_finded_main_count_key_dict[pattern] = not_finded_main_count_key_tmp_dict[model][pattern]
        not_finded_main_count_key_array.append({model: not_finded_main_count_key_dict})
    
    for model in goods_not_finded_count_tmp_dict:
        goods_not_finded_count_dict = {}
        for pattern in goods_not_finded_count_tmp_dict[model]:
            goods_not_finded_count_dict[pattern] = goods_not_finded_count_tmp_dict[model][pattern]
        goods_not_finded_count_array.append({model: goods_not_finded_count_dict})

    
    return (correctness_array, correct_data_count_array,
            incorrect_data_count_array, not_finded_main_count_key_array,
            goods_not_finded_count_array)

def transform_time_of_run_and_not_json_pattern_data_to_normal():
    time_of_run_array = []
    not_found_json_object_array = []

    for model in time_run_tmp_dict:
        time_run_dict = {}
        for pattern in time_run_tmp_dict[model]:
            add_to_time_of_run_dict_transform(time_run_dict, pattern, time_run_tmp_dict[model][pattern])
        time_of_run_array.append({model: time_run_dict})
        
    
    for model in not_found_json_tmp_dict:
        not_found_json_object_dict = {}
        for pattern in not_found_json_tmp_dict[model]:
            add_to_not_found_json_tmp_dict_transform(not_found_json_object_dict, pattern, not_found_json_tmp_dict[model][pattern])
        not_found_json_object_array.append({model: not_found_json_object_dict})
    
    return time_of_run_array, not_found_json_object_array

def call_generating_graphs_and_tables_main():
    global time_of_run_dict_tmp

    if type_of_dataset == "ticket" and not load_cpu_gpu_data:
        generate_graph("correctness")
        generate_graph("incorrect_data")
        generate_graph("not_found")
        generate_graph("goods_not_found")
    if not load_cpu_gpu_data:
        generate_graph("time_of_run")
        time_of_run_dict_tmp = time_run_dict.copy()
        generate_latex_table_and_save_to_file("time_of_run")

    if type_of_dataset == "objects" and not load_cpu_gpu_data:
        generate_bar_graph_from_data(precision_sum_dict, "precision")
        generate_bar_graph_from_data(recall_sum_dict, "recall")
    if not load_cpu_gpu_data:
        generate_bar_graph_from_data(not_found_json_dict, "not_json")
    
    if load_cpu_gpu_data:
        generate_graph("cpu_usage_main_thread")
        generate_graph("cpu_usage_peak")
        generate_graph("ram_usage_peak")
        generate_graph("gpu_usage")
        generate_graph("vram_usage")
        if not is_cpu_gpu_data_test:
            generate_graph("time_of_run_cpu_gpu")
            time_of_run_dict_tmp = cpu_gpu_data_time_diffs.copy()
            generate_latex_table_and_save_to_file("time_of_run")

def call_generating_graphs_and_tables_patterns(data_arrays_ocr, data_arrays_objects, data_arrays_objects_main):
    global correctness_dict
    global correct_data_count_dict
    global incorrect_data_count_dict
    global not_finded_main_count_key_dict
    global goods_not_finded_count_dict

    global map_dict
    global map_50_dict
    global map_75_dict
    global map_large_dict
    global mar_100_dict
    global mar_large_dict
    global time_run_dict
    global not_found_json_dict
    global model_string_for_pattern
    
    if type_of_dataset == "ticket":
        correctness_array = data_arrays_ocr[0]
        correct_data_count_array = data_arrays_ocr[1]
        incorrect_data_count_array = data_arrays_ocr[2]
        not_finded_main_count_key_array = data_arrays_ocr[3]
        goods_not_finded_count_array = data_arrays_ocr[4]

        for correctness_dict_tmp in correctness_array:
            correctness_dict.clear()
            model_string_for_pattern = next(iter(correctness_dict_tmp))
            correctness_dict = correctness_dict_tmp[model_string_for_pattern]
            generate_graph("correctness")
        
        for correct_data_count_dict_tmp in correct_data_count_array:
            correct_data_count_dict.clear()
            model_string_for_pattern = next(iter(correct_data_count_dict_tmp))
            correct_data_count_dict = correct_data_count_dict_tmp[model_string_for_pattern]
        
        for incorrect_data_count_dict_tmp in incorrect_data_count_array:
            incorrect_data_count_dict.clear()
            model_string_for_pattern = next(iter(incorrect_data_count_dict_tmp))
            incorrect_data_count_dict = incorrect_data_count_dict_tmp[model_string_for_pattern]
            generate_graph("incorrect_data")
        
        for not_finded_main_count_key_dict_tmp in not_finded_main_count_key_array:
            not_finded_main_count_key_dict.clear()
            model_string_for_pattern = next(iter(not_finded_main_count_key_dict_tmp))
            not_finded_main_count_key_dict = not_finded_main_count_key_dict_tmp[model_string_for_pattern]
            generate_graph("not_found")
        
        for goods_not_finded_count_dict_tmp in goods_not_finded_count_array:
            goods_not_finded_count_dict.clear()
            model_string_for_pattern = next(iter(goods_not_finded_count_dict_tmp))
            goods_not_finded_count_dict = goods_not_finded_count_dict_tmp[model_string_for_pattern]
            generate_graph("goods_not_found")

        time_of_run_array = data_arrays_objects_main[0]
        not_found_json_object_array = data_arrays_objects_main[1]

        for time_run_dict_tmp in time_of_run_array:
            time_run_dict.clear()
            model_string_for_pattern = next(iter(time_run_dict_tmp))
            time_run_dict = time_run_dict_tmp[model_string_for_pattern]
            generate_graph("time_of_run")

        for not_found_json_object_dict_tmp in not_found_json_object_array:
            not_found_json_object_dict.clear()
            model_string_for_pattern = next(iter(not_found_json_object_dict_tmp))
            not_found_json_dict = not_found_json_object_dict_tmp[model_string_for_pattern]
            generate_bar_graph_from_data(not_found_json_dict, "not_json")
    elif type_of_dataset == "objects":
        map_array = data_arrays_objects[0]
        map_50_array = data_arrays_objects[1]
        map_75_array = data_arrays_objects[2]
        map_large_array = data_arrays_objects[3]
        mar_100_array = data_arrays_objects[4]
        mar_large_array = data_arrays_objects[5]

        for map_dict_tmp in map_array:
            map_dict.clear()
            model = next(iter(map_dict_tmp))
            map_dict = map_dict_tmp[model]
            generate_grouped_bar_objects(map_dict, model, "MAP")

        for map_50_dict_tmp in map_50_array:
            map_50_dict.clear()
            model = next(iter(map_50_dict_tmp))
            map_50_dict = map_50_dict_tmp[model]
            generate_grouped_bar_objects(map_50_dict, model, "MAP@50")

        for map_75_dict_tmp in map_75_array:
            map_75_dict.clear()
            model = next(iter(map_75_dict_tmp))
            map_75_dict = map_75_dict_tmp[model]
            generate_grouped_bar_objects(map_75_dict, model, "MAP@75")

        for map_large_dict_tmp in map_large_array:
            map_large_dict.clear()
            model = next(iter(map_large_dict_tmp))
            map_large_dict = map_large_dict_tmp[model]
            generate_grouped_bar_objects(map_large_dict, model, "MAP_Large")

        for mar_100_dict_tmp in mar_100_array:
            mar_100_dict.clear()
            model = next(iter(mar_100_dict_tmp))
            mar_100_dict = mar_100_dict_tmp[model]
            generate_grouped_bar_objects(mar_100_dict, model, "MAR@100")

        for mar_large_dict_tmp in mar_large_array:
            mar_large_dict.clear()
            model = next(iter(mar_large_dict_tmp))
            mar_large_dict = mar_large_dict_tmp[model]
            generate_grouped_bar_objects(mar_large_dict, model, "MAR_Large")

        time_of_run_array = data_arrays_objects_main[0]
        not_found_json_object_array = data_arrays_objects_main[1]

        for time_run_dict_tmp in time_of_run_array:
            time_run_dict.clear()
            model_string_for_pattern = next(iter(time_run_dict_tmp)) 
            time_run_dict = time_run_dict_tmp[model_string_for_pattern]
            generate_graph("time_of_run")

        for not_found_json_object_dict_tmp in not_found_json_object_array:
            not_found_json_object_dict.clear()
            model = next(iter(not_found_json_object_dict_tmp))
            not_found_json_pattern_dict = not_found_json_object_dict_tmp[model]
            not_found_json_pattern_dict_tmp = {}
            for pattern in not_found_json_pattern_dict:
                not_found_json_pattern_dict_tmp[pattern] = count_number_of_types_jsons(not_found_json_pattern_dict[pattern])
            
            generate_grouped_bar_objects(not_found_json_pattern_dict_tmp, model, "count_of_data")
    else:
        print("Not found type of dataset.")
        return


def generate_all_graphs_and_tables():
    if not is_pattern_data:
        load_all_data()
        call_generating_graphs_and_tables_main()
    else:
        load_all_data_pattern()
        data_arrays_ocr = transform_ocr_pattern_data_to_normal()
        data_arrays_objects = transform_object_pattern_data_to_normal()
        data_arrays_objects_main = transform_time_of_run_and_not_json_pattern_data_to_normal()
        call_generating_graphs_and_tables_patterns(data_arrays_ocr, data_arrays_objects, data_arrays_objects_main)

def call_generating_graphs_and_tables():
    global load_cpu_gpu_data
    global is_cpu_gpu_data_test

    #main data
    set_output_dir()
    generate_all_graphs_and_tables()

    if not is_pattern_data:
        #CPU/GPU data test
        load_cpu_gpu_data = True
        is_cpu_gpu_data_test = True
        set_output_dir()
        generate_all_graphs_and_tables()

        #CPU/GPU data train
        if type_of_dataset == "objects":
            load_cpu_gpu_data = True
            is_cpu_gpu_data_test = False
            set_output_dir()
            generate_all_graphs_and_tables()

if __name__ == "__main__":
    #ticket data
    call_generating_graphs_and_tables()

    make_initial_structures()
    #objects data
    type_of_dataset = "objects"
    call_generating_graphs_and_tables()
