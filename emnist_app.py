import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Disable TensorFlow warnings and set memory growth
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Display library versions for debugging
st.sidebar.markdown("### 📋 Library Versions")
st.sidebar.text(f"TensorFlow: {tf.__version__}")
st.sidebar.text(f"Streamlit: {st.__version__}")
st.sidebar.text(f"NumPy: {np.__version__}")
st.sidebar.text(f"Python: 3.10")

# Configure GPU memory growth if available
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# App title
st.title("🧠 EMNIST Hyperparameter Tuning (Centralized Training)")
st.markdown("Train a digit recognizer on EMNIST with customizable hyperparameters.")

# Sidebar sliders
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
epochs = st.sidebar.slider("Epochs", 1, 10, 5)
batch_size = st.sidebar.slider("Batch Size", 32, 256, 64, step=32)
optimizer_name = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])

# Alternative data loading using Keras datasets
@st.cache_data
def load_mnist_data():
    """Load MNIST data as an alternative to EMNIST"""
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape to add channel dimension
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        st.info("Using MNIST dataset (EMNIST alternative)")
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

@st.cache_data
def load_emnist_data():
    """Try to load EMNIST data with tensorflow_datasets"""
    try:
        import tensorflow_datasets as tfds
        
        # Load EMNIST digits
        (ds_train, ds_test), info = tfds.load(
            "emnist/digits",
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
            download=True
        )
        
        # Convert to numpy arrays
        x_train, y_train = [], []
        for image, label in ds_train.take(10000):  # Limit for demo
            x_train.append(image.numpy())
            y_train.append(label.numpy())
        
        x_test, y_test = [], []
        for image, label in ds_test.take(2000):  # Limit for demo
            x_test.append(image.numpy())
            y_test.append(label.numpy())
        
        x_train = np.array(x_train, dtype=np.float32) / 255.0
        y_train = np.array(y_train, dtype=np.int32)
        x_test = np.array(x_test, dtype=np.float32) / 255.0
        y_test = np.array(y_test, dtype=np.int32)
        
        st.success("EMNIST data loaded successfully!")
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        st.warning(f"EMNIST loading failed: {str(e)}")
        return None, None

def create_model(input_shape=(28, 28, 1)):
    """Create a simple CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return model

def get_optimizer(optimizer_name, learning_rate):
    """Get optimizer based on name and learning rate"""
    optimizers = {
        "adam": tf.keras.optimizers.Adam(learning_rate=learning_rate),
        "sgd": tf.keras.optimizers.SGD(learning_rate=learning_rate),
        "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    }
    return optimizers.get(optimizer_name, tf.keras.optimizers.Adam(learning_rate=learning_rate))

def train_model(train_data, test_data, lr, optimizer_name, epochs, batch_size):
    """Train the model with given parameters"""
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    # Create and compile model
    model = create_model()
    optimizer = get_optimizer(optimizer_name, lr)
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Create a custom callback for Streamlit
    class StreamlitCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
            self.epoch_data = []
        
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            self.progress_bar.progress(progress)
            self.status_text.text(f'Epoch {epoch + 1}/{epochs} - Loss: {logs["loss"]:.4f} - Acc: {logs["accuracy"]:.4f}')
            
            self.epoch_data.append({
                'epoch': epoch + 1,
                'loss': logs['loss'],
                'accuracy': logs['accuracy'],
                'val_loss': logs['val_loss'],
                'val_accuracy': logs['val_accuracy']
            })
    
    callback = StreamlitCallback()
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,  # Suppress default output
        callbacks=[callback]
    )
    
    return history, callback.epoch_data

def plot_training_history(epoch_data):
    """Plot training history"""
    if not epoch_data:
        st.error("No training data to plot")
        return
    
    df = pd.DataFrame(epoch_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(df['epoch'], df['loss'], label='Training Loss', marker='o')
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(df['epoch'], df['accuracy'], label='Training Accuracy', marker='o')
    ax2.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy', marker='s')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Main app logic
def main():
    st.sidebar.markdown("### Model Configuration")
    
    # Try to load data
    with st.spinner("Loading dataset..."):
        # First try EMNIST, then fall back to MNIST
        train_data, test_data = load_emnist_data()
        
        if train_data is None:
            st.info("Falling back to MNIST dataset...")
            train_data, test_data = load_mnist_data()
        
        if train_data is None:
            st.error("Failed to load any dataset. Please check your internet connection and try again.")
            return
    
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    # Display dataset info
    st.subheader("📊 Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Train samples", len(x_train))
    with col2:
        st.metric("Test samples", len(x_test))
    with col3:
        st.metric("Image size", "28x28")
    with col4:
        st.metric("Classes", len(np.unique(y_train)))
    
    # Show sample images
    if st.checkbox("Show sample images"):
        fig, axes = plt.subplots(1, 5, figsize=(12, 3))
        for i in range(5):
            axes[i].imshow(x_train[i].squeeze(), cmap='gray')
            axes[i].set_title(f'Label: {y_train[i]}')
            axes[i].axis('off')
        st.pyplot(fig)
        plt.close()
    
    # Training section
    st.subheader("🚀 Model Training")
    
    # Display current configuration
    st.write("**Current Configuration:**")
    config_col1, config_col2 = st.columns(2)
    with config_col1:
        st.write(f"• Learning Rate: `{learning_rate}`")
        st.write(f"• Epochs: `{epochs}`")
    with config_col2:
        st.write(f"• Batch Size: `{batch_size}`")
        st.write(f"• Optimizer: `{optimizer_name}`")
    
    # Train button
    if st.button("🚀 Start Training", type="primary"):
        st.write("---")
        st.subheader("Training Progress")
        
        try:
            with st.spinner("Initializing model..."):
                history, epoch_data = train_model(
                    train_data, test_data, 
                    learning_rate, optimizer_name, 
                    epochs, batch_size
                )
            
            st.success("✅ Training completed successfully!")
            
            # Show final metrics
            final_train_acc = epoch_data[-1]['accuracy']
            final_val_acc = epoch_data[-1]['val_accuracy']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Training Accuracy", f"{final_train_acc:.3f}")
            with col2:
                st.metric("Final Validation Accuracy", f"{final_val_acc:.3f}")
            
            # Plot results
            st.subheader("📈 Training Results")
            plot_training_history(epoch_data)
            
            # Show training data table
            if st.checkbox("Show detailed training log"):
                df = pd.DataFrame(epoch_data)
                st.dataframe(df)
                
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.write("**Possible solutions:**")
            st.write("- Reduce batch size")
            st.write("- Reduce number of epochs")
            st.write("- Check your internet connection")
            st.write("- Restart the Streamlit app")

if __name__ == "__main__":
    main()
