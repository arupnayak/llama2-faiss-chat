from PIL import Image
import pytesseract

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

print(extract_text_from_image('1691224374833.png'))