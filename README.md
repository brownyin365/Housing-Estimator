# Jupy — First ML project with Jupyter and TensorFlow\n\nThis small project demonstrates training a tiny TensorFlow model on house_price_regression_dataset inside a Jupyter notebook and via a small script.\n\n## Setup (Windows PowerShell)\n\n1. Create and activate a virtual environment:\n\n```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```
\n(If your PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` first.)\n\n2. Install dependencies:\n\n```powershell
pip install -r requirements.txt
```
\n3. Start the notebook and open `notebook.ipynb`:\n\n```powershell
jupyter notebook
```
\n## Quick run (script)\n\nYou can run the provided training script (it contains the same flow as the notebook):\n\n```powershell
python train.py
```
\nNotes:\n- The notebook includes a `%pip install -r requirements.txt` cell to help install packages from the notebook kernel.\n- Training is intentionally small (low epoch count) so it runs quickly on CPU. Increase `epochs` in the notebook or `train.py` if you want a longer run.\n
