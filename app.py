import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from src.model import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model()
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("Mushroom Health Detector")

uploaded_file = st.file_uploader("Upload a mushroom image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        prob = torch.softmax(output, 1)[0]

    label = "Healthy" if predicted.item() == 0 else "Unhealthy"
    confidence = prob[predicted.item()].item()

    st.write(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.4f}")