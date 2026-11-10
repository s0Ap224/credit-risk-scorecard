"""Run the credit scoring workflow in dependency order using this interpreter."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-figures', action='store_true',
                        help='Also render Chinese figures; requires CJK fonts.')
    args = parser.parse_args()
    required = [ROOT / 'cs-training.csv', ROOT / 'Data Dictionary.xls']
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        parser.error('Download Kaggle data into the project root first. Missing: ' + ', '.join(missing))
    figure_args = [] if args.with_figures else ['--stats-only']
    steps = [
        ['split_samples.py'],
        ['audit_data.py'],
        ['feature_risk.py', *figure_args],
        ['preprocess.py'],
        ['binning_woe.py', *figure_args],
        ['fit_logistic.py'],
        ['scorecard.py'],
        ['evaluate_models.py', *figure_args],
        ['decision_thresholds.py', 'freeze'],
        ['decision_thresholds.py', 'evaluate'],
        ['score_psi.py'],
    ]
    env = {**os.environ, 'PYTHONUTF8': '1', 'MPLBACKEND': 'Agg'}
    for index, (script, *arguments) in enumerate(steps, start=1):
        print(f'[{index}/{len(steps)}] {script}', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *arguments],
                       cwd=ROOT, env=env, check=True)
    print('Completed. Outputs are in results/.')


if __name__ == '__main__':
    main()
