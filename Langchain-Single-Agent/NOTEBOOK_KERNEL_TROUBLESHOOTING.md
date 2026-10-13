# Notebook Kernel Troubleshooting

## Error

The notebook reports:

```text
ModuleNotFoundError: No module named 'langchain_community'
```

## Cause

The notebook is using a different Python interpreter from the one where the
project dependencies are installed. In this project, the packages are installed
in the Python 3.11 virtual environment at `.venv311`. If the notebook uses a
Python 3.13 kernel instead, it cannot import packages from `.venv311`.

## Fix in VS Code

1. Open `research/agent_demo.ipynb`.
2. Click the kernel selector in the notebook toolbar.
3. Select **Python 3.11 (Langchain-Single-Agent)**.
4. Restart the kernel.
5. Run the notebook cells from the top.

The project kernel uses:

```text
.venv311\Scripts\python.exe
```

## Recreate the environment if the kernel is unavailable

Open PowerShell in the `Langchain-Single-Agent` folder and run:

```powershell
py -3.11 -m venv .venv311
& ".\.venv311\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv311\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv311\Scripts\python.exe" -m ipykernel install --user --name langchain-single-agent-py311 --display-name "Python 3.11 (Langchain-Single-Agent)"
```

Select the newly registered kernel in VS Code, restart it, and run all notebook
cells. The requirements include `ipykernel` so the environment can be used as a
notebook kernel.

## Verify the notebook interpreter

Run this in a notebook cell:

```python
import sys
print(sys.executable)
```

The output should end with:

```text
Langchain-Single-Agent\.venv311\Scripts\python.exe
```

If it points to Python 3.13 or another environment, switch kernels; reinstalling
packages into the wrong interpreter will not fix the notebook.
