# Jupy — First ML project with Jupyter and TensorFlow,This small project demonstrates training a tiny TensorFlow model on House Prices inside a Jupyter notebook and via a small script.Setup (Windows PowerShell). Create and activate a virtual environment:```powershell

python -m venv .venv; .\.venv\Scripts\Activate.ps1
```
(If your PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` first.). Install dependencies:```powershell
pip install -r requirements.txt
```
Start the notebook and open `notebook.ipynb`:```powershell
jupyter notebook
```
Quick run (script)You can run the provided training script (it contains the same flow as the notebook):```powershell
python train.py
```
Notes- The notebook includes a `%pip install -r requirements.txt` cell to help install packages from the notebook kernel. - Training is intentionally small (low epoch count) so it runs quickly on CPU. Increase `epochs` in the notebook or `train.py` if you want a longer run.
