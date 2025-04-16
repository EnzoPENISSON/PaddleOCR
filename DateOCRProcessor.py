import cv2
from PIL import Image
import numpy as np
import re
from paddleocr import PaddleOCR

class DateOCRProcessor:
    def __init__(self, image_path, det_model_dir, cls_model_dir, rec_model_dir, rec_char_dict_path, use_gpu=False):
        self.image_path = image_path
        self.det_model_dir = det_model_dir
        self.cls_model_dir = cls_model_dir
        self.rec_model_dir = rec_model_dir
        self.rec_char_dict_path = rec_char_dict_path
        self.use_gpu = use_gpu

        # Initialize PaddleOCR instance
        self.ocr = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            use_gpu=self.use_gpu,
            det_model_dir=self.det_model_dir,
            rec_model_dir=self.rec_model_dir,
            cls_model_dir=self.cls_model_dir,
            rec_char_dict_path=self.rec_char_dict_path,
        )

        # Date pattern for recognition
        self.date_pattern = re.compile(r'\b(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])/\d{4}\b')

        # Character correction mapping
        self.char_map = {
            'O': '0', 'o': '0',
            'I': '1', 'l': '1',
            'Z': '2', 'z': '2',
            'S': '5', 's': '5',
            'B': '8',
            'A': '4', 'a': '0',
            'G': '6',
            'T': '7',
            'Q': '9',
        }

    def preprocess_image(self):
        """
        Read an image, convert it to grayscale, and apply Otsu's thresholding.
        Returns a PIL Image for use with PaddleOCR.
        """
        img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        upscale_factor = 2  # Adjust as needed
        upscaled_img = cv2.resize(img, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)
        if img is None:
            raise ValueError(f"Unable to load image at {self.image_path}")
        gray = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(binary)

    def correct_text(self, text):
        """
        Converts misrecognized letters in a string to their closest number forms.
        Example: 'Z2/10/2011' -> '22/10/2011'
        """
        corrected = ''.join(self.char_map.get(c, c) for c in text)

        # Convert string to a list to modify specific positions
        corrected_list = list(corrected)

        # Fix misread slashes at the expected positions (2 and 5)
        for pos in [2, 5]:
            if pos < len(corrected_list) and corrected_list[pos] in ['1', '7']:
                corrected_list[pos] = '/'

        withoutslash = ''.join(corrected_list).replace("/", "")

        # Add slashes at the expected positions (2 and 5)
        return withoutslash[:2] + '/' + withoutslash[2:4] + '/' + withoutslash[4:] if len(withoutslash) >= 8 else corrected

    def find_date(self):
        processed_image = self.preprocess_image()
        result = self.ocr.ocr(np.array(processed_image), cls=True)

        for line in result[0]:
            text = line[1][0]
            confident = line[1][1]

            match = self.date_pattern.search(text)
            if match:
                return match.group()

            corrected = self.correct_text(text)
            print("Corrected text: ", corrected)
            match = self.date_pattern.search(corrected)
            if match:
                return match.group()

        return "Aucune date trouvée ou date non reconnue"

if __name__ == "__main__":
    # Example usage
    processor = DateOCRProcessor(
        image_path="./samples/newsign2.png",
        det_model_dir="./inference/det/en_PP-OCRv3_det_slim_infer/",
        cls_model_dir="./inference/cls/ch_ppocr_mobile_v2.0_cls_infer/",
        rec_model_dir="./inference/reg/en_PP-OCRv3_rec_infer/",
        rec_char_dict_path="./ppocr/utils/en_dict.txt",
        use_gpu=False
    )

    print("------ OCR Result ------")
    print("Résultat : " + processor.find_date())
    print("------------------------")
