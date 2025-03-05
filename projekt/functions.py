import json

def get_avg_time_run(array_of_diffs):
    sum_time = 0

    for diff in array_of_diffs:
        sum_time += diff
    
    return sum_time / len(array_of_diffs)

def check_the_data(dict_model, name_of_file, path_to_correct_data):
    correct_data_counted = 0
    incorrect_data_counted = 0
    not_in_dict_counted = 0
    goods_not_counted = 0

    dict_incorrect = {}
    array_not_found = []
    array_goods_not = []

    with open(path_to_correct_data, 'r') as file:
        data = json.load(file)[name_of_file]
        
        count_of_data = data["count_of_data"]

        try:
            dict_model = json.loads(dict_model)
        except:
            return (0, 0, 0, count_of_data, 0, {}, [], [])

        try:
            company_data = dict_model["company"].lower()
            if data["company"].lower() == company_data:
                print("Company Correct")
                correct_data_counted += 1
            else:
                print("Company Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["company"] = company_data
        except:
            print("Company Not In Dict")
            not_in_dict_counted += 1
            array_not_found += ["company"]
        
        try:
            address_data = data["address"].lower()
            if data["address"].lower() == address_data:
                print("Address Correct")
                correct_data_counted += 1
            else:
                print("Address Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["address"] = address_data
        except:
            print("Address Not In Dict")
            not_in_dict_counted += 1
            array_not_found += ["address"]
        
        try:
            phone_number_data = dict_model["phone_number"]
            if data["phone_number"] == phone_number_data:
                print("Phone Number Correct")
                correct_data_counted += 1
            else:
                print("Phone Number Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["phone_number"] = phone_number_data
        except:
            print("Phone Number Not In Dict")
            not_in_dict_counted += 1
            array_not_found += ["phone_number"]
        
        try:
            server_data = dict_model["server"].lower()
            if data["server"].lower() == server_data:
                print("Server Correct")
                correct_data_counted += 1
            else:
                print("Server Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["server"] = server_data
        except:
            print("Server Not In Dict")
            not_in_dict_counted += 1
            array_not_found += ["server"]

        try:
            station_data = int(dict_model["station"])
            if data["station"] == station_data:
                print("Station Correct")
                correct_data_counted += 1
            else:
                print("Station Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["station"] = station_data
        except:
            print("Station Not In Dict")
            not_in_dict_counted += 1
            array_not_found += ["station"]
        
        try:
            order_number_data = int(dict_model["order_number"])
            if data["order_number"] == order_number_data:
                print("Order Number Correct")
                correct_data_counted += 1
            else:
                print("Order Number Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["order_number"] = order_number_data
        except:
            print("Order NUmber Not In Dict")
            not_in_dict_counted += 1
            array_not_found += ["order_number"]
        
        try:
            table_data = dict_model["table"].lower()
            if data["table"].lower() == table_data:
                print("Table Correct")
                correct_data_counted += 1
            else:
                print("Table Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["table"] = table_data
        except:
            print("Table Not In Dict")
            not_in_dict_counted += 1
            array_not_found += ["table"]
        
        try:
            guests_data = int(dict_model["guests"])
            if data["guests"] == guests_data:
                print("Guests Correct")
                correct_data_counted += 1
            else:
                print("Guests Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["guests"] = guests_data
        except:
            print("Guests Not In Dict")
            not_in_dict_counted += 1
            array_not_found += ["guests"]

        for good in data["goods"]:
            if (good in dict_model["goods"]):
                correct_data_counted += 1
                print("Good Correct")
                try:
                    amount_data = int(dict_model["goods"][good]["amount"])
                    if data["goods"][good]["amount"] == amount_data:
                        print("Amount Correct")
                        correct_data_counted += 1
                    else:
                        print("Amount Incorrect")
                        incorrect_data_counted += 1
                        dict_incorrect["amount"] = amount_data
                except:
                    print("Amount Not In Dict")
                    not_in_dict_counted += 1
                    array_not_found += ["amount"]
                
                try:
                    price_data = float(dict_model["goods"][good]["price"])
                    if data["goods"][good]["price"] == price_data:
                        print("Price Correct")
                        correct_data_counted += 1
                    else:
                        print("Price Incorrect")
                        incorrect_data_counted += 1
                        dict_incorrect["price"] = price_data
                except:
                    print("Price Not In Dict")
                    not_in_dict_counted += 1
                    array_not_found += ["price"]
            else:
                print(f"{good} Incorrect Or Not In File")
                goods_not_counted += 1
                array_goods_not += [good]
        
        try:
            subtotal_data = float(dict_model["sub_total"])
            if data["sub_total"] == subtotal_data:
                print("Subtotal Correct")
                correct_data_counted += 1
            else:
                print("SubTotal Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["subtotal"] = subtotal_data
        except:
            print("Subtotal Not In Dict")
            not_in_dict_counted += 1
            array_not_found += ["subtotal"]
        
        try:
            tax_data = float(dict_model["tax"])
            if data["tax"] == tax_data:
                print("Tax Correct")
                correct_data_counted += 1
            else:
                print("Tax Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["tax"] = tax_data
        except:
            print("Tax Not In Dict")
            not_in_dict_counted += 1
            array_not_found += ["tax"]
        
        try:
            total_data = float(dict_model["total"])
            if data["total"] == total_data:
                print("Total Correct")
                correct_data_counted += 1
            else:
                print("Total Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["total"] = total_data
        except:
            print("Total Not In Dict")
            not_in_dict_counted += 1
            array_not_found += ["total"]

        try:
            date_data = dict_model["date"]
            if data["date"] == date_data:
                print("Date Correct")
                correct_data_counted += 1
            else:
                print("Date Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["date"] = date_data
        except:
            print("Date Not In Dict")
            not_in_dict_counted += 1
            array_not_found += ["date"]
        
        try:
            time_data = dict_model["time"].lower()
            if time_data == data["time"].lower():
                print("Time Correct")
                correct_data_counted += 1
            else:
                print("Time Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["time"] = time_data
        except:
            print("Time Not In Dict")
            not_in_dict_counted += 1
            array_not_found += ["time"]

        correctness = correct_data_counted / count_of_data

        return (correctness, correct_data_counted, incorrect_data_counted, not_in_dict_counted, goods_not_counted, dict_incorrect, array_not_found, array_goods_not)
    
def save_to_file(model, type_of_data, values, incorrect_data, not_found_data, good_not_found):
    output_file_path = f"./output/{model}_{type_of_data}.txt"

    correctness = values[0]
    correct_data_counted = values[1]
    incorrect_data_counted = values[2]
    not_data_found_counted = values[3]
    good_not_found_counted = values[4]
    time_diff = values[5]
    
    with open(output_file_path, "+a") as file:
        file.write(f"{correctness};{correct_data_counted};{incorrect_data_counted};{not_data_found_counted};{good_not_found_counted};{time_diff};{incorrect_data};{not_found_data};{good_not_found}\n")