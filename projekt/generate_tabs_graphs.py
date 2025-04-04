import os
import matplotlib.pyplot as plt
import pandas as pd

output_dir = "./output/"

correctness_dict = {}
correct_data_count_dict = {}
incorrect_data_count_dict = {}
not_finded_main_count_key_dict = {}
goods_not_finded_count_dict = {}
time_run_dict = {}

def get_count_of_all_data(key):
    count_of_all_data = correct_data_count_dict[key] 
    + incorrect_data_count_dict[key] 
    + not_finded_main_count_key_dict[key] 
    + 3 * goods_not_finded_count_dict[key]
    return count_of_all_data

def generate_boxplot(tick_labels, values, y_label, type_data):
    colors = ['blue', 'green', 'red', 'purple', 'brown',
              'pink', 'gray', 'olive', 'cyan', 'maroon',
              'gold', 'lime']

    fig, ax = plt.subplots()
    ax.set_ylabel(y_label)
    ax.set_xticklabels(tick_labels)
    bplot = ax.boxplot(values, patch_artist=True)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.savefig(f"./graphs/{type_data}.png")

def generate_graph(type_of_data):
    tick_labels = []
    values = []
    y_label = ""

    if type_of_data == "correctness":
        for key in correctness_dict:
            tick_labels += [key]
            values += [correctness_dict[key]]
        y_label = "Správnost výsledku [%]"
    elif type_of_data == "correct_data":
        for key in correct_data_count_dict:
            tick_labels += [key]
            values += [correct_data_count_dict[key]]
        y_label = "Počty správných dat"
    elif type_of_data == "incorrect_data":
        for key in incorrect_data_count_dict:
            count_of_all_data = get_count_of_all_data(key)
            tick_labels += [key]
            values += [incorrect_data_count_dict[key] / count_of_all_data]
        y_label = "Poměr špatnných dat [%]"
    elif type_of_data == "not_found":
        for key in not_finded_main_count_key_dict:
            count_of_all_data = get_count_of_all_data(key)
            tick_labels += [key]
            values += [not_finded_main_count_key_dict[key] / count_of_all_data]
        y_label = "Poměr nenalezených dat [%]"
    elif type_of_data == "goods_not_found":
        for key in goods_not_finded_count_dict:
            count_of_all_data = get_count_of_all_data(key)
            data_of_goods = goods_not_finded_count_dict[key] * 3
            tick_labels += [key]
            values += [data_of_goods / count_of_all_data]
        y_label = "Poměr nenalezených zboží [%]"
    elif type_of_data == "time_of_run":
        for key in time_run_dict:
            tick_labels += [key]
            values += [time_run_dict[key]]
        y_label = "Délka běhu [s]"
    else:
        print("Not found type of data.")
        return
    
    generate_boxplot(tick_labels, values, y_label, type_of_data)


def load_all_data():
    for file in os.listdir(output_dir):
        model = file.split("_")[0]
        path_to_data = output_dir + file

        correctness_array = []
        correct_data_count_array = []
        incorrect_data_count_array = []
        not_finded_main_count_key_array = []
        goods_not_finded_count_array = []
        time_run_array = []

        with open(path_to_data, "r") as file:
            lines = file.readlines()

            for line in lines:
                array_of_values = line.replace("\n", "").split(";")
                correctness_array += [float(array_of_values[0])]
                correct_data_count_array += [int(array_of_values[1])]
                incorrect_data_count_array += [int(array_of_values[2])]
                not_finded_main_count_key_array += [int(array_of_values[3])]
                goods_not_finded_count_array += [int(array_of_values[4])]
                time_run_array += [float(array_of_values[5])]
        
        correctness_dict[model] = correctness_array
        correct_data_count_dict[model] = correct_data_count_array
        incorrect_data_count_dict[model] = incorrect_data_count_array
        not_finded_main_count_key_dict[model] = not_finded_main_count_key_array
        goods_not_finded_count_dict[model] = goods_not_finded_count_array
        time_run_dict[model] = time_run_array

if __name__ == "__main__":
    load_all_data()

    generate_graph("correctness")