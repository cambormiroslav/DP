import os
import matplotlib.pyplot as plt

type_of_dataset = "ticket"

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

graphs_dir = "./graphs/"
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)

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
precision_sum_dict = {}
recall_sum_dict = {}

cpu_gpu_data = {}
cpu_gpu_data_time_diffs = {}

time_of_run_dict_tmp = {}

def set_output_dir():
    global output_dir
    if load_cpu_gpu_data:
        if is_cpu_gpu_data_test:
            output_dir = "./test_measurement/"
        else:
            output_dir = "./train_measurement/"
    else:
        if type_of_dataset == "ticket":
            output_dir = "./output/"
        else:
            output_dir = "./output_objects/"

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
    elif load_cpu_gpu_data and not is_cpu_gpu_data_test:
        plt.savefig(f"{graphs_dir}{type_data}_{type_of_dataset}_train.svg")
    elif load_cpu_gpu_data and is_cpu_gpu_data_test:
        plt.savefig(f"{graphs_dir}{type_data}_{type_of_dataset}_test.svg")
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
    
    generate_boxplot(tick_labels, values, y_label, type_of_data)

def generate_bar_graph_from_data(dict_data, type_of_data):
    names = []
    values = []

    for key in dict_data:
        names += [key]
        values += [float(dict_data[key]) / 103]
    
    generate_bar(names, values, type_of_data)

def load_output_of_models(file_path, model_name):
    path_to_data = output_dir + file_path

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
                    if model_name in not_found_json_dict:
                        not_found_json_dict[model_name] =  not_found_json_dict[model_name] + 1
                    else:
                        not_found_json_dict[model_name] = 1
            else:
                if correct_data == 0 and incorect_data == 0 and not_finded > 0 and array_of_values[5] == "{}" and array_of_values[6] == "[]":
                    if model_name in not_found_json_dict:
                        not_found_json_dict[model_name] =  not_found_json_dict[model_name] + 1
                    else:
                        not_found_json_dict[model_name] = 1
                        
        
    correctness_dict[model_name] = correctness_array
    correct_data_count_dict[model_name] = correct_data_count_array
    incorrect_data_count_dict[model_name] = incorrect_data_count_array
    not_finded_main_count_key_dict[model_name] = not_finded_main_count_key_array
    goods_not_finded_count_dict[model_name] = goods_not_finded_count_array
    time_run_dict[model_name] = time_run_array
    precision_sum_dict[model_name] = precision_sum
    recall_sum_dict[model_name] = recall_sum

def load_output_of_models_objects(file_path, model_name, iou):
    path_to_data = output_dir + file_path

    precision_sum = 0.0
    recall_sum = 0.0
    number_of_entries = 0

    with open(path_to_data, "r") as file:
        lines = file.readlines()

        for line in lines:
            array_of_values = line.replace("\n", "").split(";")

            precision_sum += float(array_of_values[4])
            recall_sum += float(array_of_values[5])
            number_of_entries += 1
    
    if model_name not in precision_sum_dict:
        precision_sum_dict[model_name] = {iou: precision_sum / number_of_entries}
    else:
        precision_sum_dict[model_name][iou] = precision_sum / number_of_entries
    
    if model_name not in recall_sum_dict:
        recall_sum_dict[model_name] = {iou: recall_sum / number_of_entries}
    else:
        recall_sum_dict[model_name][iou] = recall_sum / number_of_entries

def load_output_of_models_objects_main(file_path, model_name):
    path_to_data = output_dir + file_path

    array_of_time_of_run = []

    with open(path_to_data, "r") as file:
        lines = file.readlines()

        for line in lines:
            time_of_run = line.replace("\n", "")
            array_of_time_of_run += [float(time_of_run)]
    
    time_run_dict[model_name] = array_of_time_of_run

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
        
def generate_all_graphs_and_tables():
    global time_of_run_dict_tmp

    load_all_data()

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

def call_generating_graphs_and_tables():
    global load_cpu_gpu_data
    global is_cpu_gpu_data_test

    #main data
    set_output_dir()
    generate_all_graphs_and_tables()

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

    #objects data
    type_of_dataset = "objects"
    call_generating_graphs_and_tables()

