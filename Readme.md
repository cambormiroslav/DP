# DP

Diplomová práce na téma Porovnání velkých multimodálních jazykových modelů a menších specializovaných modelů v úlohách optického rozpoznávání textu (OCR) a detekce objektů.

## Adresář data_for_control

Obsahuje data pro kontrolu výstupů jednotlivých modelů. Tyto data jsou veformátu JSON. Hlavní klíč je jméno souboru a hodnotou je specifikace správných dat. Obsahem jsou správná data jak pro optické rozpoznávání textu ta i pro detekci objektů.

## Adresář dataset

Obsahuje soubory dat pro testování jednotlivých modelů. Obsahem jsou dva soubory dat. První obsahuje 200 účetenek, ale pro testy bylo použito prvních 103. Také se zde vyskytuje soubor dat s testovacími soubory pro detekci objektů. Počet souborů tohoto typu je také 103.

## Adresář projekt

Obsahuje vlastní implementaci řešení diplomové práce. Struktura popsána v Readme obsažené v samotném adresáři projekt.