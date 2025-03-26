import json

"""
* Check the response characteristics.
* Check corectness of data.
* I not corrected output is: (0, 0, 0, count_of_correct_data, 0, {}, [], []).

Input: (Dictionary model as string, Name of comparing img, Path to correct data file)
Output: (Correctness, Count of correct data, Count of incorrect data, 
        Not founded data in response (main keys), Not founded number of goods,
        Dictionary of incorrect data, Array of not founded data (only keys),
        Array of not founded names goods)
"""
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
            if "company" in data:
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
            if "address" in data:
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
            if "phone_number" in data:
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
            if "server" in data:
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
            if "station" in data:
                print("Station Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["station"]
        
        try:
            order_number_data = dict_model["order_number"]
            if data["order_number"] == order_number_data:
                print("Order Number Correct")
                correct_data_counted += 1
            else:
                print("Order Number Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["order_number"] = order_number_data
        except:
            if "order_number" in data:
                print("Order Number Not In Dict")
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
            if "table" in data:
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
            if "guests" in data:
                print("Guests Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["guests"]

        try:
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
                        if "amount" in data["goods"][good]:
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
                        if "price" in data["goods"][good]:
                            print("Price Not In Dict")
                            not_in_dict_counted += 1
                            array_not_found += ["price"]
                else:
                    print(f"{good} Incorrect Or Not In File")
                    goods_not_counted += 1
                    array_goods_not += [good]
        except:
            try:
                goods_not_counted = len(data["goods"])
                for good in data["goods"]:
                    array_goods_not += [good]
                    print(f"{good} Incorrect Or Not In File")
            except:
                goods_not_counted = 0
            

        try:
            subtotal_data = float(dict_model["sub_total"])
            if data["sub_total"] == subtotal_data:
                print("Subtotal Correct")
                correct_data_counted += 1
            else:
                print("SubTotal Incorrect")
                incorrect_data_counted += 1
                dict_incorrect["sub_total"] = subtotal_data
        except:
            if "sub_total" in data:
                print("Subtotal Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["sub_total"]
        
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
            if "tax" in data:
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
            if "total" in data:
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
            if "date" in data:
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
            if "time" in data:
                print("Time Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["time"]

        if "fax_number" in data:
            try:
                fax_number_data = dict_model["fax_number"].lower()
                if fax_number_data == data["fax_number"].lower():
                    print("Fax Number Correct")
                    correct_data_counted += 1
                else:
                    print("Fax Number Incorrect")
                    incorrect_data_counted += 1
                    dict_incorrect["fax_number"] = time_data
            except:
                print("Fax Number Not In Dict")
                not_in_dict_counted += 1
                array_not_found += ["fax_number"]

        correctness = correct_data_counted / count_of_data

        return (correctness, correct_data_counted, incorrect_data_counted, not_in_dict_counted, goods_not_counted, dict_incorrect, array_not_found, array_goods_not)
"""
* Save the characteristics of model response to the file.

Input: (model name, type of data, charakteristics of data and time of run, incorrect data dict, 
        not founded data array, not founded goods)
Output: None
"""  
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