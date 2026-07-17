# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection with Grad-CAM",
    page_icon="🫁",
    layout="wide"
)

# Title and description
st.title("🫁 Chest X-Ray Pneumonia Detection")
st.markdown("""
This application uses a Custom CNN model to detect pneumonia from chest X-ray images.
The model provides Grad-CAM visualizations to explain its predictions.
""")

# Constants
IMG_SIZE = 224
NUM_CLASSES = 1
CLASS_NAMES = ['Normal', 'Pneumonia']

# Model building function
@st.cache_resource
def build_custom_cnn():
    """Build and load the trained model"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    )
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Load weights if available
    if os.path.exists('model_weights.h5'):
        model.load_weights('model_weights.h5')
    
    return model

# Grad-CAM functions
@st.cache_resource
def get_gradcam_model(model, last_conv_layer_name="conv2d_3"):
    """Create Grad-CAM model"""
    grad_model = tf.keras.Model(
        inputs=[model.input],
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )
    return grad_model

def make_gradcam_heatmap(img_array, grad_model):
    """Generate Grad-CAM heatmap"""
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    if img.max() <= 1.0:
        img = np.uint8(255 * img)
    
    return cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

def preprocess_image(image):
    """Preprocess uploaded image"""
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_array, img_batch

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    **Model**: Custom CNN with 4 convolutional layers
    
    **Visualization**: Grad-CAM (Gradient-weighted Class Activation Mapping)
    
    **Classes**:
    - Normal
    - Pneumonia
    
    **How to use**:
    1. Upload a chest X-ray image
    2. Click "Predict"
    3. View prediction and Grad-CAM visualization
    """)
    
    st.markdown("---")
    st.markdown("### Sample Images")
    if st.button("Load Sample Image"):
        st.warning("Sample images will be loaded in the main area")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray image for pneumonia detection"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

# Prediction button and results
with col2:
    st.header("Prediction Results")
    
    if uploaded_file is not None:
        if st.button("🫁 Predict", type="primary"):
            try:
                # Load model
                with st.spinner("Loading model..."):
                    model = build_custom_cnn()
                
                # Preprocess image
                with st.spinner("Processing image..."):
                    img_array, img_batch = preprocess_image(image)
                
                # Predict
                with st.spinner("Making prediction..."):
                    prediction = model.predict(img_batch, verbose=0)[0][0]
                    pred_class = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
                    confidence = prediction if prediction > 0.5 else 1 - prediction
                
                # Generate Grad-CAM
                with st.spinner("Generating Grad-CAM visualization..."):
                    grad_model = get_gradcam_model(model)
                    heatmap = make_gradcam_heatmap(img_batch, grad_model)
                    overlay = overlay_heatmap(img_array, heatmap, alpha=0.4)
                
                # Display results
                st.markdown(f"""
                ### Prediction: **{pred_class}**
                Confidence: **{confidence:.1%}**
                """)
                
                if pred_class == "PNEUMONIA":
                    st.error("⚠️ Pneumonia detected")
                else:
                    st.success("✅ Normal chest X-ray")
                
                # Display Grad-CAM
                st.subheader("Grad-CAM Visualization")
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original
                axes[0].imshow(img_array)
                axes[0].set_title("Original X-Ray")
                axes[0].axis('off')
                
                # Heatmap
                axes[1].imshow(heatmap, cmap='jet')
                axes[1].set_title("Heatmap")
                axes[1].axis('off')
                
                # Overlay
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                axes[2].imshow(overlay_rgb)
                axes[2].set_title(
                    f"{pred_class} ({confidence:.1%})",
                    color='red' if pred_class == "PNEUMONIA" else 'green'
                )
                axes[2].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.caption("Red/Yellow regions indicate areas the model focused on for prediction")
                
                # Additional info
                with st.expander("ℹ️ Model Information"):
                    st.markdown("""
                    - **Model Architecture**: Custom CNN
                    - **Input Size**: 224x224x3
                    - **Output**: Binary classification (Normal/Pneumonia)
                    - **Visualization**: Grad-CAM highlights important regions
                    """)
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Please make sure you have the model weights file (model_weights.h5) in the same directory.")
    else:
        st.info("👈 Upload an image to get started")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    Built with Streamlit, TensorFlow, and Grad-CAM
</div>
""", unsafe_allow_html=True)
