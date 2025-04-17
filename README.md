https://medium.com/@anhtuan_40207/tutorial-ocr-with-paddleocr-pp-ocr-9a4342e4d7f




python ./tools/infer/predict_system.py --image_dir="./samples/exemple/sign21.png" --det_model_dir="./inference/det/en_PP-OCRv3_det_slim_infer/" --cls_model_dir="./inference/cls/ch_ppocr_mobile_v2.0_cls_infer" --rec_model_dir="./inference/reg/en_PP-OCRv3_rec_infer/"  --rec_char_dict_path="./ppocr/utils/en_dict.txt"


Image: ./samples/exemple/sign5.png, Expected: 17/05/2024, Actual: 05/12/2020, Status: FAIL
| img_path                    | expected                                 | actual                                   | status |
| :-------------------------- | :--------------------------------------- | :--------------------------------------- | :----- |
| ./samples/newsign.png       | 16/04/2025                               | 16/04/2025                               | PASS   |
| ./samples/newsign2.png      | 12/03/2019                               | 12/03/2019                               | PASS   |
| ./samples/2011.png          | 22/10/2011                               | 22/10/2011                               | PASS   |
| ./samples/sign1.png         | 19/04/2024                               | 19/04/2024                               | PASS   |
| ./samples/exemple/sign2.png | 01/10/2024                               | 01/10/2024                               | PASS   |
| ./samples/exemple/sign3.png | Aucune date trouvée ou date non reconnue | Aucune date trouvée ou date non reconnue | PASS   |
| ./samples/exemple/sign4.png | 17/05/2024                               | 17/05/2024                               | PASS   |
| ./samples/exemple/sign5.png | 17/05/2024                               | 05/12/2020                               | FAIL   |