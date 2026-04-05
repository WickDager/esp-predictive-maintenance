import json

NOTEBOOKS = [
    'notebooks/03_LSTM_Autoencoder.ipynb',
    'notebooks/04_Transformer_Anomaly_Detection.ipynb',
    'notebooks/05_RUL_Prediction.ipynb',
    'notebooks/06_Survival_Analysis.ipynb',
]

CELL1_CODE = [
    "# -- Cell 1: Environment setup -------------------------------------------------\n",
    "import sys, os\n",
    "IN_COLAB = 'google.colab' in sys.modules\n",
    "\n",
    "if IN_COLAB:\n",
    "    # Clone the full repo (Colab only copies this notebook by default)\n",
    "    if not os.path.exists('src'):\n",
    "        !git clone https://github.com/WickDager/esp-predictive-maintenance.git\n",
    "        %cd esp-predictive-maintenance\n",
    "\n",
    "    !pip install -q torch torchvision tqdm scikit-learn\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "    SAVE_DIR = '/content/drive/MyDrive/esp_pm/checkpoints'\n",
    "else:\n",
    "    SAVE_DIR = 'checkpoints'\n",
    "\n",
    "os.makedirs(SAVE_DIR, exist_ok=True)\n",
    "os.makedirs('results', exist_ok=True)\n",
    "\n",
    "import torch\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f'Device: {DEVICE}')\n",
    "if torch.cuda.is_available():\n",
    "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
]


def fix_notebook(path):
    with open(path, 'r') as f:
        nb = json.load(f)

    # Find the first code cell (Cell 1)
    cell_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            cell_idx = i
            break

    if cell_idx is None:
        print(f"  WARNING: No code cells found in {path}")
        return

    # Replace Cell 1
    nb['cells'][cell_idx]['source'] = CELL1_CODE

    # Fix sys.path to use current directory instead of parent
    for cell in nb['cells']:
        for j, line in enumerate(cell['source']):
            if "sys.path.insert" in line and ".." in line:
                cell['source'][j] = line.replace('os.path.abspath("..")', 'os.path.abspath(".")')
            if "sys.path.insert" in line and ".." in line:
                cell['source'][j] = line.replace("os.path.abspath('..')", 'os.path.abspath(".")')

    # Fix all '../results/' to 'results/'
    for cell in nb['cells']:
        for j, line in enumerate(cell['source']):
            if '../results/' in line:
                cell['source'][j] = line.replace('../results/', 'results/')
            if '../checkpoints/' in line:
                cell['source'][j] = line.replace('../checkpoints/', 'checkpoints/')

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

    # Verify
    cell1_text = ''.join(nb['cells'][cell_idx]['source'])
    print(f"  git clone: {'YES' if 'git clone' in cell1_text else 'NO'}")
    print(f"  sys.path fix: checking...")


for nb_path in NOTEBOOKS:
    print(f"\nFixing {nb_path}...")
    fix_notebook(nb_path)

print("\nDone! All notebooks updated.")
