from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTModel,
    TrOCRConfig,
    ViTConfig,
    TrOCRForCausalLM,
)
from flask_cors import CORS
from flask import Flask, request
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# Define custom configurations
vit_config = ViTConfig(
    image_size=224,
    patch_size=16,
    num_hidden_layers=12,
    num_attention_heads=16,
    hidden_size=768
)

trocr_config = TrOCRConfig(
    vocab_size=50304,
    hidden_size=1024,
    num_hidden_layers=12,
    num_attention_heads=16,
    intermediate_size=4096,
    feature_extractor_type="TrOCRProcessor"  
)

# Initialize models
encoder = ViTModel(vit_config)
decoder = TrOCRForCausalLM(trocr_config)
model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

# Initialize processor with custom configuration
# Using DeiTImageProcessor in a custom processor setup if needed
# For demonstration, we assume TrOCRProcessor aligns with your requirements
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')

def image_to_text(word_image):
    # Convert PIL Image or ndarray to tensor and preprocess
    pixel_values = processor(images=word_image, return_tensors="pt").pixel_values
    # Generate text from image
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def segmentation(file):
    nparr = np.frombuffer(file.read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    ctrs, _ = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

    output_text = ''
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi = img_np[y:y + h, x:x + w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        output_text += '\n' + image_to_text(roi_rgb)

    return output_text

@app.route("/", methods=['POST'])
def get_images():
    if request.files and request.files['file']:
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        text = segmentation(file)
        return text
    return 'Error: File not found'

if __name__ == '__main__':
    app.run(debug=True)
