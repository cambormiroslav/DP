import re
import easyocr  # <--- ZMĚNA: import easyocr
import os
import json
import datetime
import psutil
import time
import threading

import functions

correct_data_path = "../data_for_control/dataset_correct_data.json"

reader = easyocr.Reader(['en'], gpu=True)

def extract_receipt_data(image_path):
    try:
        ocr_results = reader.readtext(image_path)
        text = "\n".join([result[1] for result in ocr_results])
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if not lines:
            raise ValueError("OCR nenašlo na obrázku žádný text.")

        
        receipt_data = {
            "source_file": os.path.basename(image_path),
            "company": lines[0],
            "address": None, "phone_number": None, "server": None,
            "order_number": None, "table": None, "guests": None,
            "station": None, "date": None, "time": None, "sub_total": None, 
            "tax": None, "total": None, "fax_number": None, "goods": {}
        }

        
        patterns = {
            'phone': re.compile(r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'),
            'fax': re.compile(r'(Fax)\s*[:.\s]*(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', re.IGNORECASE),
            'fax2': re.compile(r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\s*\(Fax\)', re.IGNORECASE),
            'date': re.compile(r'(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})'),
            'time': re.compile(r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)', re.IGNORECASE),
            'server': re.compile(r'(Server|Cashier|SvrCk|User|Empl|Serv|Employee|Server #)[:\s]+([\w\s\.]+?)\b', re.IGNORECASE),
            'order': re.compile(r'(Order|Check|Check#|Order #|Order#|ORDER#|Chk|Ticket|ORDER|Check No|Bill|Ord|ORD|Trans #|Your Order)[\s#:]+([\w-]+)', re.IGNORECASE),
            'table': re.compile(r'(Table|Tbl|TABLE|Tab|Tb#|Table No|Table #|TAB#)\s*#?([\w\s/]+)', re.IGNORECASE),
            'guests': re.compile(r'(Guests|Party|#Party|Gst|Guest|Cust|No. of Guest)[:\s]+(\d+)', re.IGNORECASE),
            'station': re.compile(r'(Station|Stat|STAT)[:\s]+(\w+)', re.IGNORECASE),
            'subtotal': re.compile(r'(Subtotal|Sub Total|SubTotal|SUB TOTAL|SUBTOTAL|Sub/Ttl|Sub-total|Items|town/MA tax|FOOD|Food|Net Total|NET TOTAL|Amount)\s*[:\s]*\$?([\d,]+\.\d{2})', re.IGNORECASE),
            'tax': re.compile(r'(Tax|Sales Tax|SALES TAX|Tax 1|TAX|State Tax|STATE TAX|StateTax|Taxes|TAXES|TXTL|Total Taxes|TAX A|Castro Valley Sales Tax)\s*\d*\s*[:\s]*\$?([\d,]+\.\d{2})', re.IGNORECASE),
            'total': re.compile(r'(Total|Take-Out Total|Balance|Balance Due|TOTAL|TOTAL DUE|Total Due|Amount Due|AmountDue|AMOUNT DUE|Grand Total|GRAND TOTAL|\*DRV THRU|TOTL|TOTAL EURO|Amt Due|AMT DUE|PAYMENT|Payment|TO-GO|Order Total|Total D1-4)\s*[:\s]*\$?([\d,]+\.\d{2})', re.IGNORECASE),
            'item': re.compile(r'^(?:(\d+)\s+)?(.*?)\s+\$?([\d,]+\.\d{2})$')
        }
        
        
        footer_keywords = ['subtotal', 'tax', 'total', 'cash', 'change', 'balance', 'tip', 'gratuity']

        
        for line in lines[1:]:
            
            item_match = patterns['item'].match(line)

            if item_match and not any(keyword in item_match.group(2).lower() for keyword in footer_keywords):
                quantity = int(item_match.group(1)) if item_match.group(1) else 1
                name = item_match.group(2).strip()
                price = float(item_match.group(3).replace(',', ''))
                receipt_data['goods'][name] = {"amount": quantity, "price": price}
                continue
            
            if not receipt_data['total'] and (m := patterns['total'].search(line)): receipt_data['total'] = float(m.group(2).replace(',', ''))
            elif not receipt_data['sub_total'] and (m := patterns['subtotal'].search(line)): receipt_data['sub_total'] = float(m.group(2).replace(',', ''))
            elif not receipt_data['tax'] and (m := patterns['tax'].search(line)): receipt_data['tax'] = float(m.group(2).replace(',', ''))
            elif not receipt_data['date'] and (m := patterns['date'].search(line)): receipt_data['date'] = m.group(1)
            elif not receipt_data['time'] and (m := patterns['time'].search(line)): receipt_data['time'] = m.group(1)
            elif not receipt_data['server'] and (m := patterns['server'].search(line)): receipt_data['server'] = m.group(2).strip()
            elif not receipt_data['order_number'] and (m := patterns['order'].search(line)): receipt_data['order_number'] = m.group(2).strip()
            elif not receipt_data['table'] and (m := patterns['table'].search(line)): receipt_data['table'] = m.group(2).strip()
            elif not receipt_data['station'] and (m := patterns['station'].search(line)): receipt_data['station'] = m.group(2).strip()
            elif not receipt_data['guests'] and (m := patterns['guests'].search(line)): receipt_data['guests'] = int(m.group(2))
            elif not receipt_data['fax_number'] and (m := patterns['fax'].search(line)): receipt_data['fax_number'] = m.group(2).strip()
            elif not receipt_data['fax_number'] and (m := patterns['fax2'].search(line)): receipt_data['fax_number'] = m.group(1).strip()
            elif not receipt_data['phone_number'] and (m := patterns['phone'].search(line)): receipt_data['phone_number'] = m.group(1)
            elif not receipt_data['address'] and re.search(r'\d+\s+[A-Za-z]', line): receipt_data['address'] = line

        return receipt_data

    except Exception as e:
        return {"source_file": os.path.basename(image_path), "error": str(e)}

def test_ocr(path):
    json_output = extract_receipt_data(path)
    
    print(json.dumps(json_output, indent=4))

def load_and_measure(dir_path, first_ticket, latest_file):
    i = first_ticket - 1
    array_of_images = os.listdir(dir_path)

    while(True):
        file = array_of_images[i]
        
        #get process id
        pid = os.getpid()
        process = psutil.Process(pid)
        #cpu and memory before test model
        process.cpu_percent(interval=None)
        mem_before = process.memory_info().rss / (1024 * 1024)

        functions.monitor_data["is_running"] = True
        monitor_thread = threading.Thread(
            target=functions.monitor_memory, 
            args=(process,),
            daemon=True #stops if main script stops
        )
        monitor_thread.start()

        start_datetime = datetime.datetime.now()

        try:
            response = extract_receipt_data(os.path.join(dir_path, file))
        finally:
            # stop thread
            functions.monitor_data["is_running"] = False
            monitor_thread.join(timeout=1.0)
        
        end_datetime = datetime.datetime.now()
        #get cpu and ram usage
        mem_after = process.memory_info().rss / (1024 * 1024)
        peak_ram_mb = functions.monitor_data["peak_rss_mb"]
        cpu_usage = process.cpu_percent(interval=None)

        peak_ram_mb = max(peak_ram_mb, mem_after) #maximum of peak RAM and final value of RAM
        ram_usage = peak_ram_mb - mem_before

        data_tuple = functions.check_the_data_ocr(response, file, correct_data_path, False)
        correctness = data_tuple[0]
        correct_data = data_tuple[1]
        incorect_data = data_tuple[2]
        not_found_data = data_tuple[3]
        good_not_found = data_tuple[4]
        dict_of_incorect = data_tuple[5]
        array_not_found = data_tuple[6]
        array_good_not_found = data_tuple[7]

        diff_datetime = end_datetime - start_datetime
        diff_datetime_seconds = diff_datetime.total_seconds()
        
        functions.save_to_file_ocr("easyocr", "ticket", [correctness, correct_data, 
                                                              incorect_data, not_found_data, 
                                                              good_not_found, diff_datetime_seconds], 
                                                              dict_of_incorect, array_not_found, 
                                                              array_good_not_found)
        functions.save_to_file_cpu_gpu("easyocr", "ticket", True, cpu_usage, ram_usage, diff_datetime_seconds)

        i += 1
        
        print("Receipt: ", i)

        if i >= latest_file:
            break

if __name__ == "__main__":
    dir_path = "../dataset/large-receipt-image-dataset-SRD/"

    load_and_measure(dir_path, 1, 103)