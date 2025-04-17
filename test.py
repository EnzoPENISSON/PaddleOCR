import cv2
from PIL import Image
import numpy as np
import re
from paddleocr import PaddleOCR
from dateutil import parser
from datetime import datetime
import concurrent.futures
import pandas as pd
from DateOCRProcessor2 import DateOCRProcessor2

dicorect = [
    {
        "img_path": "./samples/newsign.png",
        "expected": "16/04/2025"
    },
    {
        "img_path": "./samples/newsign2.png",
        "expected": "12/03/2019"
    },
    {
        "img_path": "./samples/improved_2011.png",
        "expected": "22/10/2011"
    },
    {
        "img_path": "./samples/2011.png",
        "expected": "22/10/2011"
    },
    {
        "img_path": "./samples/IMG_4759.jpeg",
        "expected": "29/03/2025"
    },
    # Same date, different formats
    {
        "img_path": "./samples/sign1.png",
        "expected": "19/04/2024"
    },
    {
        "img_path": "./samples/sign13.png",
        "expected": "19/04/2024"
    },
    {
        "img_path": "./samples/sign14.png",
        "expected": "19/04/2024"
    },
    {
        "img_path": "./samples/sign14.png",
        "expected": "19/04/2024"
    },
    # Exemples
    {
        "img_path": "./samples/exemple/sign2.png",
        "expected": "1/10/24"
    },
    {
        "img_path": "./samples/exemple/sign3.png",
        "expected": "1/10/24"
    },
    {
        "img_path": "./samples/exemple/sign4.png",
        "expected": "17/05/24"
    },
    {
        "img_path": "./samples/exemple/sign5.png",
        "expected": "17/05/24"
    },
]


# Your list of test cases
dicorect = [
    {"img_path": "./samples/newsign.png",          "expected": "16/04/2025",},
    {"img_path": "./samples/newsign2.png",         "expected": "12/03/2019"},
    {"img_path": "./samples/2011.png",             "expected": "22/10/2011"},
    {"img_path": "./samples/sign1.png",            "expected": "19/04/2024"},
    {"img_path": "./samples/exemple/sign2.png",    "expected": "01/10/2024"},
    {"img_path": "./samples/exemple/sign3.png",    "expected": "Aucune date trouvée ou date non reconnue"},
    {"img_path": "./samples/exemple/sign4.png",    "expected": "17/05/2024"},
    {"img_path": "./samples/exemple/sign5.png",    "expected": "17/05/2024"},
]

def run_test(case):
    """Run OCR on one image and compare to expected."""
    path = case["img_path"]
    expected = case["expected"]
    try:
        actual = DateOCRProcessor2().run(path)
        status = "PASS" if actual == expected else "FAIL"
    except Exception as e:
        actual = f"ERROR: {e}"
        status = "ERROR"
    return {"img_path": path, "expected": expected, "actual": actual, "status": status}

# Run loop for
results = []
for case in dicorect:
    # Run the test case
    result = run_test(case)
    # Append the result to the list
    results.append(result)
    # Print the result
    print(f"Image: {result['img_path']}, Expected: {result['expected']}, Actual: {result['actual']}, Status: {result['status']}")

# Build a DataFrame for nice summary
df = pd.DataFrame(results)

# Print per-case results
print(df.to_markdown(index=False))

# Print overall summary
total = len(df)
passes = (df.status == "PASS").sum()
fails  = (df.status == "FAIL").sum()
errors = (df.status == "ERROR").sum()
print(f"\nSummary: {passes}/{total} passed, {fails} failed, {errors} errors.")