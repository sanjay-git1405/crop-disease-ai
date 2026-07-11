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
# --- SIDEBAR MENU ---
with st.sidebar:
    st.header("🌱 About the App")
    st.write("This AI was trained using PyTorch and a ResNet18 model to instantly diagnose 15 different types of crop conditions.")
    st.write("---")
    st.write("**How to use:**")
    st.write("1. Take a clear photo of a plant leaf.")
    st.write("2. Upload it on the main screen.")
    st.write("3. Get your instant diagnosis and treatment plan!")
# --------------------
# --- NEW DOWNLOAD SCRIPT ---
# This ensures the heavy model file is downloaded from GitHub Releases
model_path = "crop_disease_model.pth"
if not os.path.exists(model_path):
    st.info("Downloading AI model for the first time... this might take a minute!")
    # PASTE YOUR COPIED GITHUB RELEASE LINK IN THE QUOTES BELOW:
    url = "https://github.com/sanjay-git1405/crop-disease-ai/releases/download/v1.0/crop_disease_model.pth" 
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
  # Replace these with your model's actual 15 categories!
    class_names = [
        'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy','Potato___healthy','Potato___Late_blight','Tomato__Target_Spot'
        ,'Tomato__Tomato_mosaic_virus','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato_Bacterial_spot'
        ,'Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot'
        ,'Tomato_Spider_mites_Two_spotted_spider_mite',
        'Potato___Early_blight', 
        'Tomato___Healthy'
        # ... keep listing all 15 of yours here ...
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
        
        # --- UPGRADE 1: Calculate Confidence Percentage ---
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence = torch.max(probabilities).item() * 100
        
        _, predicted = torch.max(outputs, 1)
        
    result = class_names[predicted.item()]
    clean_result = result.replace("___", " - ").replace("_", " ")
    
    # --- UPGRADE 2: Treatment Advice Dictionary ---
    # You can add more specific advice for your 15 classes here!
    advice_dict = {
        # Example: "Your_Exact_Class_Name": "Advice text"
        "Tomato___Early_blight": "Remove affected leaves and apply a copper-based fungicide. Avoid watering from above.",
        "Potato___Late_blight": "Apply fungicide immediately. Destroy heavily infected plants to prevent spreading.",
        "Tomato___Healthy": "Your plant looks fantastic! Keep up the good work with regular watering and sunlight."
    }
    
    # Get the advice for the predicted result, or show a default message if not listed
    treatment = advice_dict.get(result, "Monitor the plant closely. Ensure proper sunlight, drainage, and air circulation.")
    
    # --- Display the Upgraded Results ---
    st.success(f"### Diagnosis: {clean_result}")
    st.info(f"**Confidence:** {confidence:.2f}%")
    st.warning(f"**Recommended Action:** {treatment}")