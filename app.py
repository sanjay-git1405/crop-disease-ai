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

# --- SIDEBAR MENU ---
with st.sidebar:
    st.header("🌱 About the App")
    st.write("This AI is trained to detect 15 different crop diseases instantly.")
    st.write("---")
    st.write("**How to use:**")
    st.write("1. Snap a clear photo of a plant leaf.")
    st.write("2. Upload it using the main screen.")
    st.write("3. Wait for the AI diagnosis and treatment plan!")
    st.info("Tip: Ensure the leaf is well-lit and centered for the best results.")

# Main Screen Title
st.title("🌱 Crop Disease Diagnostics AI")
st.write("Upload a picture of a plant leaf, and the AI will detect if it's healthy or diseased!")

# --- MODEL DOWNLOAD SCRIPT ---
model_path = "crop_disease_model.pth"
if not os.path.exists(model_path):
    st.info("Downloading AI model for the first time... this might take a minute!")
    # Direct link to your GitHub release asset
    url = "https://github.com/sanjay-git1405/crop-disease-ai/releases/download/v1.0/crop_disease_model.pth" 
    urllib.request.urlretrieve(url, model_path)
    st.success("Download complete!")

# 2. Setup Device and Transforms
device = torch.device('cpu')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load Model 
@st.cache_resource
def load_model():
    # NOTE: Once you find your 15 real plant names, replace this list!
    class_names = [
        'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 
        'Class_6', 'Class_7', 'Class_8', 'Class_9', 'Class_10',
        'Class_11', 'Class_12', 'Class_13', 'Class_14', 'Class_15'
    ] 
    
    # Setup ResNet18
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names)) 
    
    # Load the trained weights 
    model.load_state_dict(torch.load(model_path, map_location=device))
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
        
        # Calculate Confidence Percentage
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence = torch.max(probabilities).item() * 100
        
        _, predicted = torch.max(outputs, 1)
        
    result = class_names[predicted.item()]
    
    # Format the result nicely
    clean_result = result.replace("___", " - ").replace("_", " ")
    
    # Treatment Advice Dictionary
    advice_dict = {
        "Tomato___Early_blight": "Remove affected leaves and apply a copper-based fungicide. Avoid watering from above.",
        "Potato___Late_blight": "Apply fungicide immediately. Destroy heavily infected plants to prevent spreading.",
        "Tomato___Healthy": "Your plant looks fantastic! Keep up the good work with regular watering and sunlight."
    }
    
    # Get the advice
    treatment = advice_dict.get(result, "Monitor the plant closely. Ensure proper sunlight, drainage, and air circulation.")
    
    # Display the Upgraded Results
    st.success(f"### Diagnosis: {clean_result}")
    st.info(f"**Confidence:** {confidence:.2f}%")
    st.warning(f"**Recommended Action:** {treatment}")