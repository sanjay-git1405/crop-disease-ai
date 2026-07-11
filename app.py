import streamlit as st
import torch
from torchvision import transforms, datasets
import torchvision.models as models
import torch.nn as nn
from PIL import Image

# 1. Page Setup
st.set_page_config(page_title="Crop Disease AI", page_icon="🌱")
st.title("🌱 Crop Disease Diagnostics AI")
st.write("Upload a picture of a plant leaf, and the AI will detect if it's healthy or diseased!")

# 2. Setup Device and Transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load Model (Using st.cache_resource so it only loads once)
@st.cache_resource
def load_model():
@st.cache_resource
def load_model():
    # 1. PASTE YOUR EXACT LIST HERE instead of reading the dataset folder
    class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'] # Replace this with your copied list!
    
    # 2. Setup ResNet18
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names)) 
    
    # 3. Load the trained weights (map to CPU since the cloud doesn't have your RTX 4050)
    model.load_state_dict(torch.load("crop_disease_model.pth", map_location=torch.device('cpu')))
    model.eval()

    return model, class_names

# Load the AI
try:
    model, class_names = load_model()
except Exception as e:
    st.error("Model not found! Make sure 'crop_disease_model.pth' is in the folder and training is complete.")
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