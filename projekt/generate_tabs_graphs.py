import os

output_dir = "./output/"

correctness_dict = {}
correct_data_count_dict = {}
incorrect_data_count_dict = {}
not_finded_main_count_key_dict = {}
goods_not_finded_count_dict = {}
time_run_dict = {}


def generate_tab():
    pass

def generate_boxplot(labels, values):
    pass

def generate_graph(type_of_data):
    labels = []
    values = []

    if type_of_data == "correctness":
        for key in correctness_dict:
            labels += [key]
            values += [correctness_dict[key]]
    elif type_of_data == "correct_data":
        for key in correct_data_count_dict:
            labels += [key]
            values += [correct_data_count_dict[key]]
    elif type_of_data == "incorrect_data":
        for key in incorrect_data_count_dict:
            labels += [key]
            values += [incorrect_data_count_dict[key]]
    elif type_of_data == "not_found":
        for key in not_finded_main_count_key_dict:
            labels += [key]
            values += [not_finded_main_count_key_dict[key]]
    elif type_of_data == "goods_not_found":
        for key in goods_not_finded_count_dict:
            labels += [key]
            values += [goods_not_finded_count_dict[key]]
    elif type_of_data == "time_of_run":
        for key in time_run_dict:
            labels += [key]
            values += [time_run_dict[key]]
    else:
        print("Not found type of data.")
        return
    generate_boxplot(labels, values)


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