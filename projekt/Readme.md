# DP – projekt

Projekt obsahující řešení diplomové práce.

## Hlavní komunikační moduly

Mezi hlavní komunikační moduly patří skripty openai.py, gemini.py a ollama_api.py. Tyto skripty zajišťují hlavní komuninkaci, jak dlouho tvá dostat výsledek, naštení dat z testovacích souborů dat. Načítání dat je možné po jednom souboru (obrazových dat) tak i po celým adresáři. U Ollama API a OpenAI API je potřeba převést jednotlivé obrazová data do formátu base64. U Gemini je potřeba jen definovat cestu k souboru.

## Modul functions

Tento modul obsahuje všechny funkce, které by se opakovali v hlavních modulech. Mezi ně patří kontrolní funkce, která srovnává data oproti správným datům. Po srovnání testovacích dat se spočte přesnost, počet správných i špatných výskytů dat a nenalezených dat. Špatná data se navratí také jako JSON a nenalezená data (klíče) jako pole. Následně je v tomto souboru i funkce na spočtení průměrného trvání než je k dostání výsledek z jednotlivých modelů.

## Modul generate_tabs_graphs

Tento modul bude zajišťovat generování tabulek a grafů pro výsledný dokument.

## Multimodální modely

### Velké

* ChatGPT (openai.py)
* Gemini (gemini.py)

### Malé

* LLaVa (ollama_api.py)
* BakLLaVa (ollama_api.py)
* minicpm-v (ollama_api.py)
* MobileVLM (ollama_api.py)

## Adresář output

Adresář pro změřená data.