https://medium.com/@anhtuan_40207/tutorial-ocr-with-paddleocr-pp-ocr-9a4342e4d7f




python ./tools/infer/predict_system.py --image_dir="./samples/exemple/sign21.png" --det_model_dir="./inference/det/en_PP-OCRv3_det_slim_infer/" --cls_model_dir="./inference/cls/ch_ppocr_mobile_v2.0_cls_infer" --rec_model_dir="./inference/reg/en_PP-OCRv3_rec_infer/"  --rec_char_dict_path="./ppocr/utils/en_dict.txt"
