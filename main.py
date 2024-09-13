from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from flask_cors import CORS
from flask import Flask, request
from PIL import Image
import numpy as np
import requests
import cv2

app = Flask(__name__)
CORS(app)

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

def image_to_text(word_image):

    pixel_values = processor(images=word_image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

def normalize(image):

    if image is not None:
        
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        adaptive_thresh = cv2.adaptiveThreshold(
                                grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 8)
        
        adaptive_thresh = np.mean(adaptive_thresh)

        for i in range(len(grayscale_image)):
            for j in range(len(grayscale_image[0])):
                if grayscale_image[i][j] < adaptive_thresh:
                    grayscale_image[i][j] = 255
                else:
                    grayscale_image[i][j] = 0
                    
        print(grayscale_image)
        return grayscale_image
        
    else:
        return 'Image not found'
    
def enlarge(img):
    # Load the image
    image = cv2.imread('your_image.jpg')  # Replace 'your_image.jpg' with your image file

    # Define the scale factors for enlargement
    scale_width = 2  # Scale factor for width
    scale_height = 2  # Scale factor for height

    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the new dimensions based on the scale factors
    new_width = int(original_width * scale_width)
    new_height = int(original_height * scale_height)

    # Resize the image
    enlarged_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return enlarged_image


def segmentation(file):

    nparr = np.frombuffer(file.read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # gray = enlarge(gray)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Dilation to join nearby characters into a single line
    kernel = np.ones((5, 100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by y-coordinate to get lines
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

    line_images = []  # To store line segment images
    text_images = []  # To store text images
    output_text = ''

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = img_np[y:y + h, x:x + w]

        # Append line segment images
        line_images.append(roi)

        # Convert line segment to grayscale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        output_text += '\n' + image_to_text(roi_gray)

    return output_text
        

        # Thresholding to extract text
        # ret, text_img = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY)

        # # Append text images
        # text_images.append(text_img)

    # Show line segment and text image

# Further processing or saving images can be done here with line_images and text_images lists.

            


@app.route("/", methods=['POST'])

def getImages():

    print(request)
    if request.files and request.files['file']:
        print(request.files['file'])
        file = request.files['file']

        if file.filename == '':
            return 'No selected file'   
        
        text = segmentation(file)
        print(text)
        return text

    return 'Error: File not found'     
        
if __name__ == '__main__':
    app.run(debug=True)
