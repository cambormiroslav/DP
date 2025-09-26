import re
import pytesseract
import os
import json
import datetime
from PIL import Image

def extract_receipt_data(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path), lang='eng')
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if not lines:
            raise ValueError("OCR nenašlo na obrázku žádný text.")

        
        receipt_data = {
            "source_file": os.path.basename(image_path),
            "restaurant_name": lines[0],
            "address": None, "phone_number": None, "server_name": None,
            "order_number": None, "table_number": None, "guest_count": None,
            "station": None, "date": None, "time": None, "subtotal": None, 
            "tax": None, "total": None, "items": []
        }

        
        patterns = {
            'phone': re.compile(r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'),
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
                receipt_data['items'].append({"quantity": quantity, "item_name": name, "item_price": price})
                continue
            
            if not receipt_data['total'] and (m := patterns['total'].search(line)): receipt_data['total'] = float(m.group(2).replace(',', ''))
            elif not receipt_data['subtotal'] and (m := patterns['subtotal'].search(line)): receipt_data['subtotal'] = float(m.group(2).replace(',', ''))
            elif not receipt_data['tax'] and (m := patterns['tax'].search(line)): receipt_data['tax'] = float(m.group(2).replace(',', ''))
            elif not receipt_data['date'] and (m := patterns['date'].search(line)): receipt_data['date'] = m.group(1)
            elif not receipt_data['time'] and (m := patterns['time'].search(line)): receipt_data['time'] = m.group(1)
            elif not receipt_data['server_name'] and (m := patterns['server'].search(line)): receipt_data['server_name'] = m.group(2).strip()
            elif not receipt_data['order_number'] and (m := patterns['order'].search(line)): receipt_data['order_number'] = m.group(2).strip()
            elif not receipt_data['table_number'] and (m := patterns['table'].search(line)): receipt_data['table_number'] = m.group(2).strip()
            elif not receipt_data['station'] and (m := patterns['station'].search(line)): receipt_data['station'] = m.group(2).strip()
            elif not receipt_data['guest_count'] and (m := patterns['guests'].search(line)): receipt_data['guest_count'] = int(m.group(2))
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
        start_datetime = datetime.datetime.now()
        test_ocr(os.path.join(dir_path, file))
        end_datetime = datetime.datetime.now()

        diff_datetime = end_datetime - start_datetime
        diff_datetime_seconds = diff_datetime.total_seconds()

        i += 1

        print("Receipt: ", i)

        if i >= latest_file:
            break

if __name__ == "__main__":
    dir_path = "../dataset/large-receipt-image-dataset-SRD/"

    load_and_measure(dir_path, 1, 103)