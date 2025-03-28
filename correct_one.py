from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import pandas as pd
import os
from huggingface_hub import hf_hub_download

# تحميل البيانات
info = pd.read_csv("disease_info.csv", encoding="latin1")
sup = pd.read_csv("supplement_info.csv")
info = info[info["disease_name"] != "Raspberry : Healthy"]
sup = sup[sup["disease_name"] != "Raspberry___healthy"]
info = info.reset_index(drop=True)
sup = sup.reset_index(drop=True)

# إعداد Flask
app = Flask(__name__)

# تحميل الموديل من Hugging Face
checkpoint_path = hf_hub_download(repo_id="graduationbisho/plant-diagnosis", filename="best_model_checkpoint_epoch_17_NEW.pth")
print(f"✅ Model downloaded to {checkpoint_path}")

# تحميل الموديل
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 38)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
model.to(device)
model.eval()

# تحويل الصورة
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    img = Image.open(io.BytesIO(image)).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        index = predicted.item()
    
    # 🛑 حماية من الإندكس الغلط
    if index < 0 or index >= len(info):
        return "Unknown", "No description available", "No steps available", "No supplement", "No image"
    
    disease_name = info["disease_name"].iloc[index]
    description = info["description"].iloc[index]
    steps = info["Possible Steps"].iloc[index]
    supplement_name = sup['supplement name'].iloc[index]
    supplement_image = sup['supplement image'].iloc[index]
    
    return disease_name, description, steps, supplement_name, supplement_image

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image'].read()
    name, description, steps, supplement_name, supplement_image = predict(image)
    
    return jsonify({
        "disease_name": name,
        "description": description,
        "steps": steps,
        "supplement_name": supplement_name,
        "supplement_image": supplement_image
    })

# ✅ دعم `PORT` للـ Render
port = int(os.environ.get("PORT", 8080))  
app.run(host='0.0.0.0', port=port)
