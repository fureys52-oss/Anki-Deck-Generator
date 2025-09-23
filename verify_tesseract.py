import pytesseract
from PIL import Image
import os

# If you installed Tesseract in a non-standard location, you might need this line:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    # Create a simple dummy image with text
    image = Image.new('RGB', (600, 100), color = 'white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    draw.text((10,10), "Tesseract is working correctly!", fill='black')
    image.save("test_image.png")

    # Perform OCR on the dummy image
    text = pytesseract.image_to_string(Image.open("test_image.png"))

    # Clean up the dummy image
    os.remove("test_image.png")

    if "Tesseract" in text:
        print("\n✅ SUCCESS: Pytesseract successfully connected to the Tesseract engine.")
        print(f"   > OCR Result: '{text.strip()}'")
    else:
        print("\n❌ ERROR: Pytesseract is installed, but could not get a valid response from the Tesseract engine.")
        print("   > Please ensure Tesseract is installed and its location is in your system's PATH.")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: Could not run pytesseract. This likely means the Tesseract engine is not installed or not in the PATH.")
    print(f"   > Full Error: {e}")