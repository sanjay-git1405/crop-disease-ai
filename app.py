import streamlit as st
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import os
import urllib.request

# 1. Page Setup
st.set_page_config(page_title="Crop Disease AI", page_icon="🌱")
st.title("🌱 Crop Disease Diagnostics AI")
st.write("Upload a picture of a plant leaf, and the AI will detect if it's healthy or diseased!")

# --- NEW DOWNLOAD SCRIPT ---
# This ensures the heavy model file is downloaded from GitHub Releases
model_path = "crop_disease_model.pth"
if not os.path.exists(model_path):
    st.info("Downloading AI model for the first time... this might take a minute!")
    # PASTE YOUR COPIED GITHUB RELEASE LINK IN THE QUOTES BELOW:
    url = "PASTE_YOUR_COPIED_LINK_HERE" 
    urllib.request.urlretrieve(url, model_path)
    st.success("Download complete!")
# ---------------------------

# 2. Setup Device and Transforms
# We force the CPU device here so it works perfectly on the free Streamlit Cloud
device = torch.device('cpu')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load Model 
# st.cache_resource is the recommended way to cache global resources like ML models so it doesn't reload on every interaction.
@st.cache_resource
def load_model():
    # REPLACE THIS LIST with your exact 15 class names
    class_names = [
        'Plant_1', 'Plant_2', 'Plant_3', 'Plant_4', 'Plant_5', 
        'Plant_6', 'Plant_7', 'Plant_8', 'Plant_9', 'Plant_10',
        'Plant_11', 'Plant_12', 'Plant_13', 'Plant_14', 'Plant_15'
    ] 
    
    # Setup ResNet18
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names)) 
    
    # Load the trained weights 
    model.load_state_dict(torch.load("crop_disease_model.pth", map_location=device))
    model.eval()
    
    return model, class_names

# Load the AI safely
try:
    model, class_names = load_model()
except Exception as e:
    st.error(f"Model error: {e}. Make sure your download link is correct and class names match.")
    st.stop()

# 4. Create the File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Leaf", use_column_width=True)
    
    st.write("🧠 **AI is analyzing...**")
    
    # Process image and predict
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        
    result = class_names[predicted.item()]
    
    # Format the result nicely (e.g., changing "Potato___Early_blight" to "Potato - Early blight")
    clean_result = result.replace("___", " - ").replace("_", " ")
    
    # Display the final diagnosis!
    st.success(f"### Diagnosis: {clean_result}")