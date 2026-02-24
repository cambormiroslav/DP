# DP – projekt

Projekt obsahující řešení diplomové práce.

## Moduly

### Hlavní komunikační moduly

Mezi hlavní komunikační moduly patří skripty `easyOCR.py`, `gemini.py`, `openai.py`, `ollama_api.py`, `pytorch_models.py`, `tesseract.py`. Tyto skripty zajišťují hlavní komuninkaci s modely, zjištění, jak dlouho trvá dostat výsledek a další jiné systémové zdroje, načtení dat z testovacích souborů dat. Načítání dat je možné po jednom souboru (obrazových dat) tak i načíst celý adresář najednou. U Ollama API a OpenAI API je potřeba převést jednotlivá obrazová data do formátu base64. U Gemini je potřeba jen definovat cestu k souboru.

### Modul functions

Tento modul obsahuje všechny funkce, které by se opakovali v hlavních modulech. Mezi ně patří kontrolní funkce, která srovnává data oproti správným datům. Po srovnání testovacích dat se spočte přesnost, počet správných i špatných výskytů dat a nenalezených dat. Špatná data se navrátí také jako JSON (klíč je jméno nenalezeného data a hodnota je zkoumaná hodnota) a nenalezená data (klíče) jako pole. Dále lze tu najít funkce pro ukládání dat do souboru (jednotlivých naměřených metrik). Tyto data jsou navrácená pro testy OCR metod. Pro porovnání hodnot detekce objektů se používají metriky mAP a recall. Tyto data jsou ukládána do adresáře `output` v případě OCR a `output_objects` v případě detekce objektů. Zde také probíhá výpočet hodnot spotřeby systémových zdrojů.

### Modul generate_graphs_and_tables

Zajišťuje generování grafů. Počty navrácených dat ve špatném formátu, mAP a recall jsou generovány grafy typu `chart` a ostatní grafy jsou typu `boxplot`. Tento modul dokáže také generovat LaTeX tabulky z některých metrik.

### Modul generate_data_for_control_objects

Transformuje XML anotace ve formátu PascalVOC do formátu JSON pro kontrolu správnosti optické detekce objektů jednotlivých modelů.

### Modul test_patterns

Generuje data pro výběr nejlepšího textového zadání pro jednotlivé velké multimodální modely.

## Multimodální modely

### Velké

* BakLLaVa (ollama_api.py)
* Gemini (gemini.py)
* Gemma3 (ollama_api.py)
* Granite3.2-vision (ollama_api.py)
* LLaVa (ollama_api.py)
* ChatGPT (openai.py)
* minicpm-v (ollama_api.py)
* MobileVLM (ollama_api.py)
* Mistral-Small (ollama_api.py)

### Malé

* EasyOCR (easyOCR.py)
* FasterRCNN (pytorch_models.py)
* MaskRCNN (pytorch_models.py)
* RetinaNet (pytorch_models.py)
* Tesseract (tesseract.py)
* Yolo11 (yolo.py)

## Adresáře

### Adresář graphs

* Obsahuje grafy k metrikám naměřených během testů OCR.

### Adresář graphs_objects

* Obsahuje grafy k metrikám naměřených během testů optické detekce objektů.

### Adresář graphs_patterns

* Obsahuje grafy pro výběr textového zadání pro velké multimodální modely u technologie OCR.

### Adresář graphs_patterns_objects

* Obsahuje grafy pro výběr textového zadání pro velké multimodální modely u optické detekce objektů.

### Adresář large_of_models

* Obsahuje LaTeX tabulky na téma využití systémových zdrojů u jednotlivých modelů.

### Adresář output

* Jsou zde změřená data pro testování optického rozpoznávání textu.

### Adresář output_objects

* Jsou zde změřená data pro testování detekce objektů.

### Adresář output_pattern_test

* Obsahuje data naměřená během testů pro výběr textových zadání pro OCR.

### Adresář output_pattern_test_objects

* Obsahuje data naměřená během testů pro výběr textových zadání pro optickou detekci objektů.

### Adresář tables

* Obsahuje tabulky výpočetními zdroji pro OCR i detekci objektů.

### Adresář tables_patterns

* Obsahuje tabulky pro výběr textového zadání technologií OCR a detekci objektů.

### Adresář test_measurement

* Jsou zde naměřená data využití systémových prostředků při testování modelů.

### Adresář train_measurement

* Jsou zde naměřená data využití systémových prostředků při trénování modelů.