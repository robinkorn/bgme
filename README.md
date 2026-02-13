![scikit](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)![numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)![python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
---

# Bachelorarbeit: Boosted Gaussian Mixture Ensemble
> Von **Robin Korn**

---

## Informationen

**Python-Version:** 3.12.xx  

**Setup:**
```bash
# Virtuelle Umgebung erstellen
py -3.12 -m venv .venv

# Umgebung aktivieren
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\Activate      # Windows PowerShell

# Lokales Package installieren
pip install -e .
```
---

## Ordnerstruktur

- **data/**
Enthält alle datenbezogenen Dateien. Das ServerMachineDataset sollte hier unter *data/ServerMachineDataset* liegen.

- **src/**
Beinhaltet den gesamten Quellcode und die Projektlogik.
*src/data* enthählt Analyse- und Ladeskripte
*src/eval* enthählt die Skripte für die Evaluation, Plots und F1 Scores
*src/models* enthählt das BGME mit Basisklasse und Online-Addon
*src/sim* enthählt den Datensimulator

Einige Ordner könnten zusätzlich von Skripten erstellt werden um Ergebnisse abzuspeichern.

---

## Lizenz

MIT License
