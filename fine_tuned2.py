import optuna
import pandas as pd
from DateOCRProcessor2 import DateOCRProcessor2

#pip install optuna optuna-dashboard
#optuna-dashboard sqlite:///ocr_tuning.db


TEST_CASES = [
    {"img_path": "./samples/newsign.png",      "expected": "16/04/2025"},
    {"img_path": "./samples/newsign2.png",     "expected": "12/03/2019"},
    {"img_path": "./samples/2011.png",         "expected": "22/10/2011"},
    {"img_path": "./samples/sign1.png",        "expected": "19/04/2024"},
    {"img_path": "./samples/exemple/sign2.png","expected": "01/10/2024"},
    {"img_path": "./samples/exemple/sign3.png","expected": "Aucune date trouvée ou date non reconnue"},
    {"img_path": "./samples/exemple/sign4.png","expected": "17/05/2024"},
    {"img_path": "./samples/exemple/sign5.png","expected": "17/05/2024"},
]

def objective(trial):
    processor = DateOCRProcessor2(
        upscale_factor = trial.suggest_float("upscale_factor", 1.0, 5.0),
        denoise_h = trial.suggest_int("denoise_h", 5, 25),
        clahe_clip = trial.suggest_float("clahe_clip", 1.0, 10.0),
        clahe_grid = trial.suggest_categorical("clahe_grid", [(4, 4), (8, 8), (16, 16), (32, 32)]),
        threshold_block = trial.suggest_int("threshold_block", 11, 101, step=2),
        threshold_C = trial.suggest_int("threshold_C", 0, 25),
    )

    passed = 0
    for case in TEST_CASES:
        try:
            result = processor.run(case["img_path"])
            if result == case["expected"]:
                passed += 1
        except:
            pass
    return passed

if __name__ == "__main__":
    storage = "sqlite:///ocr_tuning.db"
    study = optuna.create_study(direction="maximize", study_name="ocr_date_tuning", storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=100)
    print("Best params:", study.best_params)
    print("Best score:", study.best_value)