import cv2
from PIL import Image
import numpy as np
import re
from paddleocr import PaddleOCR

import cv2
from PIL import Image
import numpy as np
import re
from paddleocr import PaddleOCR

class DateOCRProcessor:
    def __init__(self):
        self.image_path = None
        self.image_array = None  # Ajout pour l'image directe
        self.det_model_dir = "./inference/en_PP-OCRv3_det_slim_infer/"
        self.cls_model_dir = "./inference/ch_ppocr_mobile_v2.0_cls_infer/"
        self.rec_model_dir = "./inference/en_PP-OCRv3_rec_infer/"
        self.rec_char_dict_path = "./en_dict.txt"
        self.use_gpu = False

        self.ocr = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            use_gpu=self.use_gpu,
            det_model_dir=self.det_model_dir,
            rec_model_dir=self.rec_model_dir,
            cls_model_dir=self.cls_model_dir,
            rec_char_dict_path=self.rec_char_dict_path,
        )

        self.date_pattern = re.compile(r'\b(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])/\d{4}\b')

        self.char_map = {
            'O': '0', 'o': '0', '.': '0',
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
        Enhance image for OCR: grayscale, denoise, contrast stretch, adaptive threshold.
        Returns a PIL Image.
        """
        if self.image_array is not None:
            img = self.image_array
        elif self.image_path is not None:
            print(self.image_path)
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        else:
            raise ValueError("No image provided for OCR")

        if img is None:
            raise ValueError("Image loading failed")

        # Step 1: Upscale
        upscale_factor = 2
        img = cv2.resize(img, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 3: Denoise (optional but helps with handwriting)
        gray = cv2.fastNlMeansDenoising(gray, h=30)

        # Step 4: Contrast enhancement using histogram equalization
        gray = cv2.equalizeHist(gray)

        # Step 5: Adaptive thresholding (better than Otsu for uneven lighting)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Invert if text is dark on light background
            blockSize=31,  # Must be odd and >1
            C=10  # Smaller C makes thresholding more aggressive
        )

        # Optional: Morphological operations to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return Image.fromarray(binary)

    def correct_text(self, text):
        corrected = ''.join(self.char_map.get(c, c) for c in text)
        corrected_list = list(corrected)

        for pos in [2, 5]:
            if pos < len(corrected_list) and corrected_list[pos] in ['1', '7']:
                corrected_list[pos] = '/'

        withoutslash = ''.join(corrected_list).replace("/", "")
        return withoutslash[:2] + '/' + withoutslash[2:4] + '/' + withoutslash[4:] if len(withoutslash) >= 8 else corrected

    def find_date(self):
        processed_image = self.preprocess_image()
        result = self.ocr.ocr(np.array(processed_image), cls=True)

        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            match = self.date_pattern.search(text)
            if match:
                return match.group()

            corrected = self.correct_text(text)
            print("Corrected text: ", corrected)
            match = self.date_pattern.search(corrected)
            if match:
                return match.group()

        return "Aucune date trouvée ou date non reconnue"

    def run(self, image_input):
        """
        Run OCR on either an image path (str) or a NumPy image (array).
        """
        if isinstance(image_input, str):
            self.image_path = image_input
            self.image_array = None
        elif isinstance(image_input, np.ndarray):
            self.image_array = image_input
            self.image_path = None
        else:
            raise ValueError("image_input must be a file path or a NumPy array")

        return self.find_date()




# print(DateOCRProcessor().run("../sign2.png"))
