import json
import re

NOTEBOOKS = [
    'notebooks/03_LSTM_Autoencoder.ipynb',
    'notebooks/04_Transformer_Anomaly_Detection.ipynb',
    'notebooks/05_RUL_Prediction.ipynb',
    'notebooks/06_Survival_Analysis.ipynb',
    'notebooks/07_Model_Evaluation_and_SHAP.ipynb',
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
    "\n",
    "    # Mount Drive (needed if you want to load real data from Drive)\n",
    "    DRIVE_MOUNTED = False\n",
    "    try:\n",
    "        from google.colab import drive\n",
    "        drive.mount('/content/drive')\n",
    "        DRIVE_MOUNTED = True\n",
    "        print('Google Drive mounted.')\n",
    "    except Exception as e:\n",
    "        print(f'Drive mount skipped/failed: {e}. Using local-only mode.')\n",
    "\n",
    "    # Always save checkpoints locally (avoids Drive I/O issues during training)\n",
    "    SAVE_DIR = '/content/checkpoints'\n",
    "else:\n",
    "    DRIVE_MOUNTED = False\n",
    "    SAVE_DIR = 'checkpoints'\n",
    "\n",
    "os.makedirs(SAVE_DIR, exist_ok=True)\n",
    "os.makedirs('results', exist_ok=True)\n",
    "print(f'Checkpoints will be saved to: {SAVE_DIR}')\n",
    "\n",
    "import torch\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f'Device: {DEVICE}')\n",
    "if torch.cuda.is_available():\n",
    "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
]

# Improved configs: smaller model, more regularization, real data default, step_size=2
LSTM_CONFIG = [
    "# ── Cell 3: Hyperparameters ─\n",
    "# ──  Tune these!\n",
    "CONFIG = {\n",
    "    # Data\n",
    "    'window_size':    50,     # timesteps per input window\n",
    "    'step_size':       2,     # sliding window step (reduced overlap)\n",
    "    'val_split':    0.15,\n",
    "    'test_split':   0.15,\n",
    "\n",
    "    # Model\n",
    "    'hidden_size':    64,     # reduced from 128 to prevent memorization\n",
    "    'num_layers':      2,\n",
    "    'latent_size':    32,\n",
    "    'dropout':       0.5,     # increased from 0.3 for regularization\n",
    "\n",
    "    # Training\n",
    "    'batch_size':    128,\n",
    "    'num_epochs':    100,\n",
    "    'lr':           1e-3,\n",
    "    'weight_decay': 1e-3,     # increased from 1e-4 for stronger L2\n",
    "    'patience':       20,     # increased from 15\n",
    "    'gradient_clip': 1.0,\n",
    "\n",
    "    # Anomaly detection\n",
    "    'threshold_pct': 95,      # calibration percentile on normal windows\n",
    "    'mc_samples':     50,     # MC Dropout forward passes\n",
    "}\n",
    "\n",
    "print('CONFIG:')\n",
    "for k, v in CONFIG.items():\n",
    "    print(f'  {k:20s}: {v}')\n",
]

LSTM_DATA_CELL = [
    "# ── Cell 4: Load & prepare data ─\n",
    "# ──  CHOOSE YOUR DATA SOURCE\n",
    "# Option A: Synthetic (generated on the fly)\n",
    "# Option B: Real Pump Sensor CSV from Google Drive\n",
    "\n",
    "USE_REAL_DATA = True   # → True to load from Drive (default)\n",
    "\n",
    "if USE_REAL_DATA:\n",
    "    # ── Real Pump Sensor Data from Drive ─\n",
    "    # Expected CSV: sensor.csv from Kaggle pump-sensor-data\n",
    "    # Has columns: timestamp, sensor_00, ..., sensor_51, machine_status\n",
    "    DRIVE_CSV_PATH = '/content/drive/MyDrive/pump_sensor.csv'  # → change if needed\n",
    "    print(f'Loading real data from {DRIVE_CSV_PATH}...')\n",
    "\n",
    "    raw_df = pd.read_csv(DRIVE_CSV_PATH, parse_dates=['timestamp'])\n",
    "    raw_df = raw_df.sort_values('timestamp').reset_index(drop=True)\n",
    "\n",
    "    # Detect sensor columns (sensor_XX pattern)\n",
    "    SENSOR_COLS = sorted([c for c in raw_df.columns if c.startswith('sensor_')])\n",
    "    print(f'Found {len(SENSOR_COLS)} sensor columns')\n",
    "\n",
    "    # Forward-fill missing values, then fill remaining with 0\n",
    "    raw_df[SENSOR_COLS] = raw_df[SENSOR_COLS].ffill().bfill()\n",
    "\n",
    "    # Binary failure label\n",
    "    status_map = {'NORMAL': 0, 'RECOVERING': 0, 'BROKEN': 1}\n",
    "    raw_df['failure'] = raw_df['machine_status'].map(status_map).fillna(0).astype(int)\n",
    "    print(f'Raw data: {len(raw_df):,} rows, {len(SENSOR_COLS)} sensors')\n",
    "    print(f'Failure rate: {raw_df[\"failure\"].mean()*100:.2f}%')\n",
    "\n",
    "else:\n",
    "    # ── Synthetic Data ─\n",
    "    from src.data.synthetic_generator import generate_esp_dataset, SYNTHETIC_SENSOR_COLS\n",
    "    print('Generating synthetic ESP data...')\n",
    "    raw_df = generate_esp_dataset(n_wells=40, timesteps_per_well=4000, random_seed=42)\n",
    "    SENSOR_COLS = SYNTHETIC_SENSOR_COLS\n",
    "    raw_df['failure'] = (raw_df['machine_status'] == 'BROKEN').astype(int)\n",
    "    print(f'Raw data: {len(raw_df):,} rows, {len(SENSOR_COLS)} sensors')\n",
    "\n",
    "# Build sliding windows\n",
    "from src.data.loader import _sliding_window, _compute_rul, _split_and_scale\n",
    "\n",
    "X_raw = raw_df[SENSOR_COLS].values.astype(np.float32)\n",
    "y_raw = raw_df['failure'].values.astype(np.float32)\n",
    "rul_raw = _compute_rul(y_raw)\n",
    "\n",
    "X_w, y_w, rul_w = _sliding_window(\n",
    "    X_raw, y_raw, rul_raw,\n",
    "    CONFIG['window_size'], CONFIG['step_size']\n",
    ")\n",
    "print(f'Windows: {X_w.shape}, failures: {y_w.sum():.0f} ({y_w.mean()*100:.2f}%)')\n",
    "\n",
    "data = _split_and_scale(\n",
    "    X_w, y_w, rul_w, SENSOR_COLS,\n",
    "    val_split=CONFIG['val_split'],\n",
    "    test_split=CONFIG['test_split'],\n",
    "    random_seed=42\n",
    ")\n",
]

DRIVE_SYNC_CELL = [
    "# -- Save results to Google Drive -------------------------------------------\n",
    "import sys, os, shutil\n",
    "\n",
    "if 'google.colab' not in sys.modules:\n",
    "    print('Not running in Colab. No Drive sync needed.')\n",
    "else:\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "\n",
    "    DRIVE_DIR = '/content/drive/MyDrive/esp_pm'\n",
    "    os.makedirs(f'{DRIVE_DIR}/checkpoints', exist_ok=True)\n",
    "    os.makedirs(f'{DRIVE_DIR}/results', exist_ok=True)\n",
    "\n",
    "    # Copy checkpoints\n",
    "    if os.path.exists('checkpoints'):\n",
    "        for f in os.listdir('checkpoints'):\n",
    "            src = os.path.join('checkpoints', f)\n",
    "            dst = f'{DRIVE_DIR}/checkpoints/{f}'\n",
    "            shutil.copy2(src, dst)\n",
    "            print(f'  Copied: checkpoints/{f}')\n",
    "\n",
    "    # Copy results\n",
    "    if os.path.exists('results'):\n",
    "        for f in os.listdir('results'):\n",
    "            src = os.path.join('results', f)\n",
    "            dst = f'{DRIVE_DIR}/results/{f}'\n",
    "            shutil.copy2(src, dst)\n",
    "            print(f'  Copied: results/{f}')\n",
    "\n",
    "    # Copy training log\n",
    "    if os.path.exists(f'{SAVE_DIR}/training_log.csv'):\n",
    "        shutil.copy2(f'{SAVE_DIR}/training_log.csv', f'{DRIVE_DIR}/checkpoints/training_log.csv')\n",
    "        print(f'  Copied: training_log.csv')\n",
    "\n",
    "    print(f'\\nAll files synced to: {DRIVE_DIR}')\n",
]


def fix_mojibake(text):
    """Fix common mojibake patterns."""
    
    # Pattern 1: Fix sequences like "─њЏпёЏ" or "─њпё" that should be "──"
    text = re.sub(r'\u2500[\u0452\u040f\u043f\u0451]+[\u0452\u040f\u043f\u0451]*', '\u2500\u2500', text)
    
    # Pattern 1.5: Fix "─→→" which should be just "→"
    text = text.replace('\u2500\u2192\u2192', '\u2192')
    text = text.replace('\u2500\u2192', '\u2192')
    
    # Pattern 2: Fix "→ђ" should be just "→"
    text = text.replace('\u2192\u0452', '\u2192')
    
    # Pattern 3: Remove standalone mojibake comment markers
    for ch in ['\u0452', '\u040f', '\u043f', '\u0451', '\u045a', '\u0453', '\u041f', '\u0433', '\u041e', '\u0406']:
        text = text.replace(ch, '')
    
    # Pattern 4: Fix common broken box-drawing sequences
    box_fixes = {
        '\u0432\u201e\u0402': '\u2500',
        '\u0432\u201d\u201a': '\u2502',
        '\u0432\u201d\u0153': '\u251c',
        '\u0432\u201d\u00a4': '\u2524',
        '\u0432\u201d\u00ac': '\u252c',
        '\u0432\u201d\u00b4': '\u2534',
        '\u0432\u201d\u00bc': '\u253c',
        '\u0432\u2022\u0161': '\u2554',
        '\u0432\u2022\u2014': '\u2557',
        '\u0432\u2022\u0097': '\u255d',
        '\u0432\u2022\u00a6': '\u2566',
        '\u0432\u2022\u00a9': '\u2569',
        '\u0432\u2022\u00ac': '\u256c',
    }
    
    for broken, correct in box_fixes.items():
        text = text.replace(broken, correct)
    
    # Pattern 5: Fix arrows
    arrow_fixes = {
        '\u0432\u2020\u2019': '\u2192',
        '\u0432\u2020\u201d': '\u2193',
        '\u0432\u2020\u0091': '\u2191',
        '\u0432\u2020\u201c': '\u2190',
    }
    
    for broken, correct in arrow_fixes.items():
        text = text.replace(broken, correct)
    
    return text


def fix_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
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

    # Notebook 03 (LSTM): Also replace CONFIG and data loading cells
    cells_fixed = 0
    if '03_LSTM' in path:
        for i, cell in enumerate(nb['cells']):
            src = ''.join(cell['source'])
            # Replace CONFIG cell (has 'hidden_size': 128)
            if cell['cell_type'] == 'code' and "'hidden_size':   128" in src:
                nb['cells'][i]['source'] = LSTM_CONFIG
                cells_fixed += 1
            # Replace data loading cell (has USE_REAL_DATA = False)
            if cell['cell_type'] == 'code' and 'USE_REAL_DATA = False' in src:
                nb['cells'][i]['source'] = LSTM_DATA_CELL
                cells_fixed += 1

    # Fix all cell contents (mojibake + sys.path + paths)
    cells_fixed_total = cells_fixed
    for cell in nb['cells']:
        original_source = cell['source'][:]
        # Fix mojibake in all lines
        for j, line in enumerate(cell['source']):
            cell['source'][j] = fix_mojibake(line)
        
        # Fix sys.path to use current directory instead of parent
        for j, line in enumerate(cell['source']):
            if "sys.path.insert" in line and ".." in line:
                cell['source'][j] = line.replace('os.path.abspath("..")', 'os.path.abspath(".")')
                cell['source'][j] = line.replace("os.path.abspath('..')", 'os.path.abspath(".")')
        
        # Fix all '../results/' to 'results/'
        for j, line in enumerate(cell['source']):
            if '../results/' in line:
                cell['source'][j] = line.replace('../results/', 'results/')
            if '../checkpoints/' in line:
                cell['source'][j] = line.replace('../checkpoints/', 'checkpoints/')
        
        if original_source != cell['source']:
            cells_fixed_total += 1

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    # Add Drive sync cell at the end (if not already present)
    has_sync_cell = any(
        'Save results to Google Drive' in ''.join(cell.get('source', ''))
        for cell in nb['cells']
    )
    if not has_sync_cell:
        nb['cells'].append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': DRIVE_SYNC_CELL,
        })
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print('  Added Drive sync cell at the end.')

    # Verify
    cell1_text = ''.join(nb['cells'][cell_idx]['source'])
    print(f"  git clone: {'YES' if 'git clone' in cell1_text else 'NO'}")
    print(f"  Drive mount: {'YES' if 'drive.mount' in cell1_text else 'NO'}")
    print(f"  Cells modified: {cells_fixed_total}")
    
    # Check for remaining mojibake
    all_text = ''.join([''.join(cell['source']) for cell in nb['cells']])
    cyrillic_count = len(re.findall(r'[\u0400-\u04FF]', all_text))
    if cyrillic_count > 0:
        print(f"  WARNING: {cyrillic_count} Cyrillic characters still present")
        matches = re.findall(r'[\u0400-\u04FF]', all_text)
        print(f"    Chars: {set(matches)}")
    else:
        print(f"  All mojibake fixed!")


for nb_path in NOTEBOOKS:
    print(f"\nFixing {nb_path}...")
    fix_notebook(nb_path)

print("\nDone! All notebooks updated.")
