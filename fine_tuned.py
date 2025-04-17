import itertools
import concurrent.futures
import pandas as pd
from DateOCRProcessor2 import DateOCRProcessor2

# Define your test cases
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

# Parameter search space
SEARCH_SPACE = {
    'upscale_factor': [1.0, 1.5, 2.0, 2.5],
    'denoise_h': [5, 7, 9, 11],
    'clahe_clip': [1.0, 1.5, 2.1, 3.0],
    'clahe_grid': [(8,8), (16,16)],
    'threshold_block': [31, 41, 51],
    'threshold_C': [5, 9, 13]
}

# Generate all combinations
def generate_param_grid(search_space):
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

# Evaluate a single parameter set
def evaluate_params(params):
    processor = DateOCRProcessor2(**params)
    results = []
    for case in TEST_CASES:
        try:
            actual = processor.run(case['img_path'])
            status = (actual == case['expected'])
        except Exception:
            status = False
        results.append(status)
    score = sum(results)  # number of passed tests
    return {'params': params, 'score': score}

# Run grid search
def tune_parameters(n_jobs=4):
    best = None
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(evaluate_params, p): p for p in generate_param_grid(SEARCH_SPACE)}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if best is None or result['score'] > best['score']:
                best = result
    return best

# Main execution
if __name__ == '__main__':
    best_result = tune_parameters()
    print("Best parameter set:", best_result['params'])
    print("Passed {} out of {} tests".format(best_result['score'], len(TEST_CASES)))
