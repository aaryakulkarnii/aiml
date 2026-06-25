import json, sys, os

notebooks = [
    r'c:\Users\aarya\Downloads\aiml\notebooks\01_data_exploration.ipynb',
    r'c:\Users\aarya\Downloads\aiml\notebooks\02_data_cleaning.ipynb',
    r'c:\Users\aarya\Downloads\aiml\notebooks\03_feature_engineering.ipynb',
    r'c:\Users\aarya\Downloads\aiml\notebooks\04_modeling_xgboost.ipynb',
    r'c:\Users\aarya\Downloads\aiml\notebooks\05_modeling_rf_lgbm.ipynb',
    r'c:\Users\aarya\Downloads\aiml\notebooks\06_shap_analysis.ipynb',
    r'c:\Users\aarya\Downloads\aiml\notebooks\07_fairness_evaluation.ipynb',
]

for nb_path in notebooks:
    print(f'\n========== {os.path.basename(nb_path)} ==========')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, cell in enumerate(nb['cells']):
        cell_type = cell['cell_type']
        src = ''.join(cell['source'])
        outputs = []
        if cell_type == 'code':
            for o in cell.get('outputs', []):
                otype = o.get('output_type', '')
                if otype == 'stream':
                    text = o.get('text', '')
                    if isinstance(text, list): text = ''.join(text)
                    outputs.append(text)
                elif otype in ('execute_result', 'display_data'):
                    data = o.get('data', {})
                    text = data.get('text/plain', '')
                    if isinstance(text, list): text = ''.join(text)
                    outputs.append(text)
        print(f'--- Cell {i} [{cell_type}] ---')
        print(src[:1200])
        if outputs:
            combined = ''.join(outputs)
            print('OUTPUT:', combined[:1000])
        print()
