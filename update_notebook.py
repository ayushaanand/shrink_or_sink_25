import json

with open('colab_search.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == 'aa01':
        cell['source'] = [
            '# 🔬 Shrink or Sink — Architecture Search Notebook\n',
            '> **Runtime → Change runtime type → T4 GPU** before running!\n',
            '\n',
            'This notebook runs the full search pipeline:\n',
            '1. Install dependencies\n',
            '2. Clone the repo\n',
            '3. Show the ranked architecture search space\n',
            '4. Train the Ultimate Teacher (ResNet-50 + Pseudo-Labeling)\n',
            '5. Run binary search to find smallest model ≥ 85%\n',
            '6. Inspect and save results'
        ]
    elif cell.get('id') == 'aa02':
        cell['source'] = []
    elif cell.get('id') == 'aa05':
        cell['source'] = [
            '!rm -rf Shrink_or_Sink\n',
            '!git clone https://github.com/ayushaanand/Shrink_or_Sink.git\n',
            '%cd Shrink_or_Sink\n',
            '!pip install torch torchvision tqdm -q\n'
        ]
    elif cell.get('id') == 'aa06':
        cell['source'] = ['## 📊 Step 2 — Define Search Bounds\n']
    elif cell.get('id') == 'aa07':
        cell['source'] = [
            '# The search operates between lower and upper channel bounds.\n',
            '# It probes the midpoint between lo and hi until convergence.\n',
            'from dynamic_model import DynamicNet, size_mb, param_count\n',
            '\n',
            'lo = [8, 16, 32, 64]\n',
            'hi = [64, 128, 256, 256]\n',
            '\n',
            'print(f"Search Space Bounds:\\n")\n',
            'print(f"  lo: {lo} -> {size_mb(DynamicNet(lo)):.4f} MB ({param_count(DynamicNet(lo)):,} params)")\n',
            'print(f"  hi: {hi} -> {size_mb(DynamicNet(hi)):.4f} MB ({param_count(DynamicNet(hi)):,} params)")\n'
        ]
    elif cell.get('id') == 'aa10':
        cell['source'] = [
            '# ── Colab Resumable Teacher Training ────────────────────────────────\n',
            'import os\n',
            "TEACHER_PATH = f'{DRIVE_DIR}/teacher.pth'\n",
            '\n',
            '# The script will automatically look for teacher_latest.pth in {DRIVE_DIR} to resume!\n',
            'if os.path.exists(TEACHER_PATH):\n',
            "    print('✅ Teacher already finalized and saved. Skipping.')\n",
            'else:\n',
            '    !python teacher_train.py \\\n',
            '        --data /content/data \\\n',
            '        --out {TEACHER_PATH} \\\n',
            '        --checkpoint {DRIVE_DIR}/teacher_latest.pth\n'
        ]
    elif cell.get('id') == 'aa11':
        cell['source'] = [
            '## 🛑 STOP HERE FOR NOW\n',
            '\n',
            '> Since we are focusing exclusively on training the **Ultimate Teacher** (200 Epochs), you do not need to run the Search phase yet.\n',
            '> \n',
            '> You can just leave this Colab tab open, let the cell above run for the next few hours, and check back periodically. It will autosave `teacher_latest.pth` to your Google Drive every epoch.'
        ]
    elif cell.get('id') == 'aa08':
        cell['source'] = [
            '## 🎓 Step 3 — Train the Ultimate Teacher (ResNet-50)\n',
            'The teacher is **unconstrained in size** — it is never submitted.\n',
            'Its only job is to provide soft-label knowledge to our tiny student.\n',
            '\n',
            'This will run for 200 epochs across 3 phases (Burn-in, Pseudo-Labeling, Mastery).\n',
            '> *Tip: It saves to Drive every single epoch. If Colab crashes, just run this cell again to resume!*\n'
        ]
    elif cell.get('id') == 'aa12':
        cell['source'] = [
            "STUDENT_OUT = f'{DRIVE_DIR}/best_student.pth'\n",
            "\n",
            "import os\n",
            "if not os.path.exists(TEACHER_PATH):\n",
            "    print('🛑 STOP: Teacher is not finished training yet! Wait for Phase 3 to complete.')\n",
            "else:\n",
            "    !python search.py \\\n",
            "        --data /content/data \\\n",
            "        --teacher {TEACHER_PATH} \\\n",
            "        --lo 8 16 32 64 \\\n",
            "        --hi 64 128 256 256 \\\n",
            "        --proxy-epochs 20 \\\n",
            "        --full-epochs 100 \\\n",
            "        --proxy-thresh 0.65 \\\n",
            "        --target-acc 0.85 \\\n",
            "        --out {STUDENT_OUT}\n"
        ]
    elif cell.get('id') == 'aa14':
        cell['source'] = [
            "import json\n",
            "\n",
            "with open('search_results.json') as f:\n",
            "    log = json.load(f)\n",
            "\n",
            "print(f\"{'Iter':<6} {'Config':<30} {'MB':>8} {'Proxy':>8} {'Full':>8}  {'Verdict'}\")\n",
            "print('─' * 85)\n",
            "for r in log:\n",
            "    full = f\"{r['full_acc']:.4f}\" if r['full_acc'] is not None else '   —  '\n",
            "    print(f\"{r['iteration']:<6} {str(r['config']):<30} \"\n",
            "          f\"{r['mb']:>8.4f} {r['proxy_acc']:>8.4f} {full:>8}   {r['verdict']}\")\n"
        ]

with open('colab_search.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
