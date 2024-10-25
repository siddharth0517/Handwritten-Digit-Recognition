import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Load the trained CNN model
model = tf.keras.models.load_model('my_model.keras')

# Function to preprocess image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(image) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1)
    return img_array

# Streamlit app layout
st.title('MNIST Digit Classifier')

st.write('Draw a digit or upload an image for the model to predict.')

# Drawing canvas for the user to draw a digit
canvas_result = st_canvas(
    fill_color="black",  # Drawing color
    stroke_width=10,     # Stroke width
    stroke_color="white",  # Drawing in white
    background_color="black",  # Background black
    height=150, width=150,  # Canvas size (should be square)
    drawing_mode="freedraw",  # Drawing mode
    key="canvas"
)

# If there's a drawn image, preprocess and predict
if canvas_result.image_data is not None:
    # Convert the canvas result into an image
    drawn_image = Image.fromarray(np.uint8(canvas_result.image_data))
    
    # Preprocess the drawn image
    preprocessed_image = preprocess_image(drawn_image)
    
    # Predict the digit using the model
    prediction = model.predict(preprocessed_image)
    predicted_digit = np.argmax(prediction)
    
    # Display the predicted digit
    st.write(f"Predicted Digit: **{predicted_digit}**")

# Option to upload an image
st.write("Or upload an image below:")

# Upload the image file
uploaded_file = st.file_uploader("Choose an image...", type="png")

# If an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the uploaded image
    preprocessed_image = preprocess_image(image)
    
    # Predict the digit
    prediction = model.predict(preprocessed_image)
    predicted_digit = np.argmax(prediction)
    
    # Display the prediction result
    st.write(f"Predicted Digit: **{predicted_digit}**")
