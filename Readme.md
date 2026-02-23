# DP

Diplomová práce na téma Porovnání velkých multimodálních jazykových modelů a menších specializovaných modelů v úlohách optického rozpoznávání textu (OCR) a detekce objektů.

## Adresář data_for_control

Obsahuje data pro kontrolu výstupů jednotlivých modelů.  Jsou zde dva soubory. První ([./data_for_control/dataset_correct_data.json](./data_for_control/dataset_correct_data.json)) je pro účely kontroly optické rozpoznávání textu a druhý (([./data_for_control/dataset_objects_correct_data.json](./data_for_control/dataset_objects_correct_data.json))) je pro účely optické detekce objektů. Tyto data jsou ve formátu JSON. Hlavní klíč je jméno souboru a hodnotou je specifikace správných dat.

## Adresář dataset

Obsahuje soubory dat pro testování jednotlivých modelů. Obsahem jsou dva soubory dat. Ten první ([./dataset/large-receipt-image-dataset-SRD/](./dataset/large-receipt-image-dataset-SRD/)) je určen pro kontrolu správnosti optického rozpoznávání textu. Obsahem je 103 účtenek. Druhý ([./dataset/yolo_dataset/](./dataset/yolo_dataset/)) obsahuje data pro trénování tak i pro testování modelu. Tyto data jsou uzpůsobeny pro trénování modelu YOLO11. Obsahem jsou především už zmíněná data pro trénování modelu tich je 520 ([./dataset/yolo_dataset/train/](./dataset/yolo_dataset/train/)). Nadále tento soubor dat obsahuje testovací data ([./dataset/yolo_dataset/test/](./dataset/yolo_dataset/test/)) tich je 100 a zbylých 133 dat je pro validaci učení ([./dataset/yolo_dataset/valid/](./dataset/yolo_dataset/valid/)). Také zde lze nalézt adresář [./dataset/coco_annotations/](./dataset/coco_annotations/) zde jsou obsaženy anotace pro modely využívající COCO formát pro učení detekce objektů. Také zde je adresář [./dataset/objects/](./dataset/objects/), ve kterém jsou vytaženy testovací data.

## Adresář projekt

Obsahuje vlastní implementaci řešení diplomové práce. Struktura popsána v Readme obsažené v samotném adresáři projekt.