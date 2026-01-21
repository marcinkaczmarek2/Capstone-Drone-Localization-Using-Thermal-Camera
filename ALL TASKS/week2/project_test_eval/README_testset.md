\# Mini Testset – Detekcja dronów na termowizji



\## Cel

Stworzenie małego zestawu testowego + systemu ewaluacji IoU, precision, recall, F1.



\## Zawartość

\- obrazki: `dataset\_test/images/`

\- etykiety YOLO: `dataset\_test/labels/`

\- skrypt konwersji YOLO → JSON: `scripts/convert\_labels\_to\_json.py`

\- skrypt ewaluacji: `scripts/evaluate.py`



\## Workflow

1\. Umieść obrazy w `dataset\_test/images/`

2\. Oznacz bounding boxy w `dataset\_test/labels/`

3\. Uruchom:

&nbsp;  ```bash

&nbsp;  python scripts/convert\_labels\_to\_json.py



