import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import subprocess

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

IMG_SIZE = 224

@st.cache_resource
def download_and_train():
    """Download dataset and train model on first run"""
    
    data_path = "chest_xray_data/chest_xray"
    
    # Download dataset if not exists
    if not os.path.exists(data_path):
        with st.spinner("📥 Downloading dataset... This may take 2-3 minutes"):
            os.environ["KAGGLE_API_TOKEN"] = "KGAT_5b435f7615b49cc51191f5ab984c36d2"
            subprocess.run(["kaggle", "datasets", "download", "-d", "paultimothymooney/chest-xray-pneumonia"], 
                          capture_output=True)
            subprocess.run(["unzip", "-q", "chest-xray-pneumonia.zip", "-d", "chest_xray_data"])
    
    # Build model
    with st.spinner("🧠 Building CNN model..."):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.callbacks import EarlyStopping
        
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
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Load and train
    with st.spinner("🎯 Training model... This may take 5-7 minutes"):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            f"{data_path}/train",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=32,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        val_generator = train_datagen.flow_from_directory(
            f"{data_path}/train",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=32,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=5,
            callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
            verbose=0
        )
    
    return model

def make_gradcam_heatmap(img_array, model):
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv = layer
                break
        if last_conv is None:
            return None
        
        @tf.function
        def compute_gradcam(inputs):
            with tf.GradientTape() as tape:
                x = inputs
                conv_output = None
                for layer in model.layers:
                    x = layer(x)
                    if layer == last_conv:
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
    except:
        return None

def overlay_heatmap(img, heatmap, alpha=0.4):
    if heatmap is None:
        return None
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    if img.max() <= 1.0:
        img = np.uint8(255 * img)
    return cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

def predict_image(model, image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch, verbose=0)[0][0]
    pred_class = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return pred_class, confidence, img_array, img_batch

def main():
    st.title("🫁 Pneumonia Detection from Chest X-Ray")
    st.markdown("Upload a chest X-ray image for AI-powered pneumonia detection")
    
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        **First time:** The app will download dataset and train model (5-10 mins).
        **After that:** Instant predictions!
        """)
        st.warning("⚠️ **Medical Disclaimer:** Demonstration only")
    
    # Load model
    if 'model' not in st.session_state:
        model = download_and_train()
        st.session_state.model = model
        st.success("✅ Model ready!")
    
    # Upload
    uploaded_file = st.file_uploader(
        "📤 Choose a chest X-ray image",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded X-Ray", use_column_width=True)
        
        with st.spinner("🧠 Analyzing..."):
            pred_class, confidence, img_array, img_batch = predict_image(
                st.session_state.model, image
            )
            heatmap = make_gradcam_heatmap(img_batch, st.session_state.model)
            overlay = overlay_heatmap(img_array, heatmap)
        
        st.markdown("---")
        st.markdown("## 📊 Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", pred_class)
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        with col3:
            status = "⚠️" if pred_class == "PNEUMONIA" else "✅"
            st.metric("Status", f"{status}")
        
        if overlay is not None:
            st.markdown("## 🔍 Grad-CAM Visualization")
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(img_array)
            axes[0].set_title("Original X-Ray")
            axes[0].axis('off')
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            axes[1].imshow(overlay_rgb)
            axes[1].set_title(f"{pred_class} ({confidence:.1%})", 
                             color='red' if pred_class == "PNEUMONIA" else 'green')
            axes[1].axis('off')
            plt.tight_layout()
            st.pyplot(fig)
        
        if pred_class == "PNEUMONIA":
            st.error("⚠️ **Consult a healthcare professional immediately!**")
        else:
            st.success("✅ No pneumonia detected")

if __name__ == "__main__":
    main()
