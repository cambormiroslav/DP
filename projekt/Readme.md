# DP – projekt

Projekt obsahující řešení diplomové práce.

## Hlavní komunikační moduly

Mezi hlavní komunikační moduly patří skripty `openai.py`, `gemini.py` a `ollama_api.py`. Tyto skripty zajišťují hlavní komuninkaci s modely, zjištění jak dlouho trvá dostat výsledek, načtení dat z testovacích souborů dat. Načítání dat je možné po jednom souboru (obrazových dat) tak i načíst celý adresář najednou. U Ollama API a OpenAI API je potřeba převést jednotlivá obrazová data do formátu base64. U Gemini je potřeba jen definovat cestu k souboru.

## Modul functions

Tento modul obsahuje všechny funkce, které by se opakovali v hlavních modulech. Mezi ně patří kontrolní funkce, která srovnává data oproti správným datům. Po srovnání testovacích dat se spočte přesnost, počet správných i špatných výskytů dat a nenalezených dat. Špatná data se navratí také jako JSON (klíč je jméno nenalezeného data a hodnota je zkoumaná hodnota) a nenalezená data (klíče) jako pole. Dále lze tu najít funkce pro ukládání dat do souboru (jednotlivých naměřených metrik). Tyto data jsou ukládány do adresáře `output` a `output_objects`

## Modul generate_graphs

Zajišťuje generování grafů. Počty navrácených dat ve špatném formátu jsou generovány grafy typu `bar chart` a ostatní grafy jsou typu `boxplot`.

## Multimodální modely

### Velké

* ChatGPT (openai.py)
* Gemini (gemini.py)

### Malé

* BakLLaVa (ollama_api.py)
* Granite3.2-vision (ollama_api.py)
* LLaVa (ollama_api.py)
* minicpm-v (ollama_api.py)
* MobileVLM (ollama_api.py)
* Mistral-Small (ollama_api.py)

## Adresář output

Adresář pro změřená data pro testování optického rozpoznávání textu.

## Adresář output_objects

Adresář pro změřená data pro testování detekce objektů.