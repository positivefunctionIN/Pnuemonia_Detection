import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detection from Chest X-Ray",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
IMG_SIZE = 224
MODEL_PATH = 'pneumonia_model.h5'  # Change this to your model file name

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .pneumonia {
        background-color: #ffcccc;
        border: 2px solid #ff0000;
    }
    .normal {
        background-color: #ccffcc;
        border: 2px solid #00cc00;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model with caching
@st.cache_resource
def load_model():
    """Load the trained pneumonia detection model."""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file '{MODEL_PATH}' not found! Please upload the model file.")
            return None
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Grad-CAM functions
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_3"):
    """Generate Grad-CAM heatmap for model interpretability."""
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        
        @tf.function
        def compute_gradcam(inputs):
            with tf.GradientTape() as tape:
                x = inputs
                conv_output = None
                for layer in model.layers:
                    x = layer(x)
                    if layer == last_conv_layer:
                        tape.watch(x)
                        conv_output = x
                loss = x[0][0]
            
            grads = tape.gradient(loss, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_output = conv_output[0]
            heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
            
            return heatmap
        
        return compute_gradcam(img_array).numpy()
    except Exception as e:
        st.warning(f"⚠️ Grad-CAM visualization not available: {str(e)}")
        return None

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on the original image."""
    if heatmap is None:
        return None
    
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert image to uint8 if needed
    if img.max() <= 1.0:
        img = np.uint8(255 * img)
    
    # Overlay heatmap on image
    return cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

def preprocess_image(image):
    """Preprocess the uploaded image."""
    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    return img_array

def predict_image(model, image):
    """Make prediction on the image."""
    # Preprocess
    img_array = preprocess_image(image)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_batch, verbose=0)[0][0]
    pred_class = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return pred_class, confidence, img_array, img_batch

def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 Pneumonia Detection from Chest X-Ray</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 1.1rem; margin-bottom: 2rem;'>
    Upload a chest X-ray image and get instant AI-powered pneumonia detection with visual explanations.
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        **How it works:**
        - Upload a chest X-ray image (JPG, PNG)
        - The CNN model analyzes the image
        - Get prediction with confidence score
        - View Grad-CAM heatmap showing what the model focused on
        
        **Model Details:**
        - Custom CNN architecture
        - Trained on Chest X-Ray Pneumonia dataset
        - Test accuracy: ~90%
        - Binary classification: Normal vs Pneumonia
        """)
        
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="Adjust the confidence threshold for predictions"
        )
        
        show_heatmap = st.checkbox("Show Grad-CAM Heatmap", value=True)
        
        st.header("📋 Disclaimer")
        st.warning("""
        ⚠️ **Medical Disclaimer:** This is a demonstration tool only.
        Always consult a qualified healthcare professional for medical diagnosis.
        """)
    
    # Main content area
    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-Ray Image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a chest X-ray image for pneumonia detection"
    )
    
    if uploaded_file is not None:
        # Load model
        with st.spinner("🔄 Loading model..."):
            model = load_model()
        
        if model is None:
            st.error("❌ Model not loaded. Please check the model file.")
            return
        
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded_file, caption="📷 Uploaded Chest X-Ray", use_column_width=True)
        
        # Process image
        image = Image.open(uploaded_file)
        
        # Make prediction
        with st.spinner("🧠 Analyzing the X-ray..."):
            pred_class, confidence, img_array, img_batch = predict_image(model, image)
            
            # Generate Grad-CAM heatmap if enabled
            heatmap = None
            overlay = None
            if show_heatmap:
                heatmap = make_gradcam_heatmap(img_batch, model)
                overlay = overlay_heatmap(img_array, heatmap, alpha=0.4)
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Results")
        
        # Create metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Prediction box
            pred_color = "pneumonia" if pred_class == "PNEUMONIA" else "normal"
            pred_icon = "⚠️" if pred_class == "PNEUMONIA" else "✅"
            st.markdown(f"""
            <div class="prediction-box {pred_color}">
                <h3>{pred_icon} Prediction</h3>
                <h2 style="margin:0;">{pred_class}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Confidence
            st.metric(
                label="🎯 Confidence Score",
                value=f"{confidence:.1%}",
                delta="High confidence" if confidence > confidence_threshold else "Low confidence",
                delta_color="normal" if confidence > confidence_threshold else "inverse"
            )
        
        with col3:
            # Status
            status = "⚠️ Requires Medical Attention" if pred_class == "PNEUMONIA" else "✅ No Pneumonia Detected"
            status_color = "red" if pred_class == "PNEUMONIA" else "green"
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <h3 style="color: {status_color}; margin:0;">{status}</h3>
                <p style="color: gray; font-size: 0.9rem;">{confidence:.1%} confidence</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        if show_heatmap and overlay is not None and heatmap is not None:
            st.markdown("## 🔍 Visual Explanation")
            st.markdown("The Grad-CAM heatmap shows which regions of the X-ray influenced the model's decision:")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(img_array)
            axes[0].set_title("Original X-Ray", fontweight='bold', fontsize=14)
            axes[0].axis('off')
            
            # Heatmap
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title("Grad-CAM Heatmap", fontweight='bold', fontsize=14)
            axes[1].axis('off')
            
            # Overlay
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            axes[2].imshow(overlay_rgb)
            axes[2].set_title(
                f"{'PNEUMONIA' if pred_class == 'PNEUMONIA' else 'NORMAL'} ({confidence:.1%})",
                fontweight='bold',
                fontsize=14,
                color='red' if pred_class == "PNEUMONIA" else 'green'
            )
            axes[2].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Legend
            st.markdown("""
            <div style="display: flex; justify-content: center; gap: 2rem; margin: 1rem 0;">
                <span><span style="color: red;">■</span> Red: High importance</span>
                <span><span style="color: orange;">■</span> Orange: Medium importance</span>
                <span><span style="color: blue;">■</span> Blue: Low importance</span>
            </div>
            """, unsafe_allow_html=True)
        elif show_heatmap:
            st.warning("⚠️ Grad-CAM visualization not available for this model.")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Analyze Another Image", use_container_width=True):
                st.rerun()
        
        # Medical disclaimer
        if pred_class == "PNEUMONIA":
            st.error("""
            ### ⚠️ URGENT: Consult a Healthcare Professional
            This AI prediction suggests **PNEUMONIA**. Please consult a qualified healthcare 
            professional immediately for proper diagnosis and treatment. This tool is for 
            demonstration purposes only and should not replace medical advice.
            """)
        else:
            st.success("""
            ### ✅ No Pneumonia Detected
            The model did not detect signs of pneumonia. However, always consult a healthcare 
            professional for proper diagnosis. This tool is for demonstration purposes only.
            """)
    
    else:
        # Placeholder when no image uploaded
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f0f2f6; border-radius: 10px;">
            <h3>📤 Upload a Chest X-Ray Image</h3>
            <p style="color: gray;">Click the upload button above to get started.</p>
            <p style="color: gray; font-size: 0.9rem;">Supported formats: JPG, PNG, BMP, TIFF</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample images info
        st.markdown("""
        ### 📝 Instructions
        1. Click the "Upload Chest X-Ray Image" button above
        2. Select a chest X-ray image from your computer
        3. Wait for the AI model to analyze the image
        4. View the prediction results and visual explanation
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8rem;'>
        Built with ❤️ using TensorFlow and Streamlit | Version 1.0.0
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
