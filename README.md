# Mortgage default prediction - pilot (Freddie Mac SFLLD sample 2022-2026)

1. Download sample_2022..sample_2026 from Freddie Mac Clarity and unzip into data/.
2. Download MORTGAGE30US and state unemployment (XXUR) monthly series from FRED; merge into out/fred_monthly.csv.
3. python build_dataset.py data out
4. python run_experiments.py out out/fred_monthly.csv 10
5. python run_additional.py out out/fred_monthly.csv RF
6. python make_figures.py out figures

Seeds: 42 (Optuna, bootstrap), tree refits 42/43/44. Raw data are not included (Freddie Mac terms of use).
