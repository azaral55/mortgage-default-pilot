set -e
cd /home/claude/mortgage_pilot
python3 -I run_experiments.py out out/fred_monthly.csv 10 > out/run_main.log 2>&1
BEST=$(python3 -c "import json;r=json.load(open('out/results_main.json'))['models'];print(max([m for m in r if m!='LR'],key=lambda m:r[m]['validation_auc']))")
echo "best tree model by validation AUC: $BEST" >> out/run_main.log
python3 -I run_additional.py out out/fred_monthly.csv $BEST > out/run_additional.log 2>&1
echo PIPELINE_DONE
