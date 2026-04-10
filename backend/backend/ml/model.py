import os
import io
import asyncio
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def load_mnist_data():
    """Load MNIST data as an alternative to EMNIST"""
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        # Limit the size for faster demo
        return (x_train[:10000], y_train[:10000]), (x_test[:2000], y_test[:2000])
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        return None, None

def load_emnist_data():
    """Try to load EMNIST data with tensorflow_datasets"""
    try:
        import tensorflow_datasets as tfds
        ds_train, ds_test = tfds.load(
            "emnist/digits",
            split=["train", "test"],
            as_supervised=True,
            download=True
        )
        x_train, y_train = [], []
        # Limit to 10000 for fast demo training
        for image, label in ds_train.take(10000):
            x_train.append(image.numpy())
            y_train.append(label.numpy())
        
        x_test, y_test = [], []
        for image, label in ds_test.take(2000):
            x_test.append(image.numpy())
            y_test.append(label.numpy())
            
        x_train = np.array(x_train, dtype=np.float32) / 255.0
        y_train = np.array(y_train, dtype=np.int32)
        x_test = np.array(x_test, dtype=np.float32) / 255.0
        y_test = np.array(y_test, dtype=np.int32)
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        print(f"Error loading EMNIST: {e}")
        return None, None

def load_custom_data(dataset_id):
    """Load custom dataset from temp_uploads directory"""
    try:
        data_dir = os.path.join("temp_uploads", dataset_id)
        if not os.path.exists(data_dir):
            return None, None
            
        batch_size = 32
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            color_mode="grayscale",
            image_size=(28, 28),
            batch_size=batch_size
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            color_mode="grayscale",
            image_size=(28, 28),
            batch_size=batch_size
        )
        
        x_train, y_train = [], []
        for images, labels in train_ds:
            x_train.append(images.numpy())
            y_train.append(labels.numpy())
            
        x_test, y_test = [], []
        for images, labels in val_ds:
            x_test.append(images.numpy())
            y_test.append(labels.numpy())
            
        if not x_train:
            return None, None
            
        x_train = np.concatenate(x_train, axis=0) / 255.0
        y_train = np.concatenate(y_train, axis=0)
        x_test = np.concatenate(x_test, axis=0) / 255.0
        y_test = np.concatenate(y_test, axis=0)
        
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        print(f"Error loading custom dataset: {e}")
        return None, None

def create_model(input_shape=(28, 28, 1), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

def get_optimizer(optimizer_name, learning_rate):
    optimizers = {
        "adam": tf.keras.optimizers.Adam(learning_rate=learning_rate),
        "sgd": tf.keras.optimizers.SGD(learning_rate=learning_rate),
        "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    }
    return optimizers.get(optimizer_name, tf.keras.optimizers.Adam(learning_rate=learning_rate))

class WebSockCallback(tf.keras.callbacks.Callback):
    def __init__(self, async_queue, loop):
        super().__init__()
        self.async_queue = async_queue
        self.loop = loop

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        data = {
            "type": "epoch_end",
            "epoch": epoch + 1,
            "loss": float(logs.get('loss', 0)),
            "accuracy": float(logs.get('accuracy', 0)),
            "val_loss": float(logs.get('val_loss', 0)),
            "val_accuracy": float(logs.get('val_accuracy', 0))
        }
        # Schedule the coroutine to put data into the queue in the asyncio loop
        asyncio.run_coroutine_threadsafe(self.async_queue.put(data), self.loop)

def train_model_sync(dataset_name, lr, optimizer_name, epochs, batch_size, async_queue, loop):
    if dataset_name.lower() == "emnist":
        train_data, test_data = load_emnist_data()
        if train_data is None:
            train_data, test_data = load_mnist_data()
    elif dataset_name.lower() == "mnist":
        train_data, test_data = load_mnist_data()
    else:
        train_data, test_data = load_custom_data(dataset_name)
        
    if train_data is None:
         asyncio.run_coroutine_threadsafe(async_queue.put({"type": "error", "message": "Failed to load data"}), loop)
         return
         
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    num_classes = len(np.unique(y_train))
    model = create_model(num_classes=num_classes)
    optimizer = get_optimizer(optimizer_name, lr)
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    cb = WebSockCallback(async_queue, loop)
    
    try:
        model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[cb]
        )
        
        test_predictions = model.predict(x_test, verbose=0)
        test_pred_classes = np.argmax(test_predictions, axis=1)
        cm = tf.math.confusion_matrix(y_test, test_pred_classes, num_classes=num_classes)
        cm_list = cm.numpy().tolist()
        
        asyncio.run_coroutine_threadsafe(async_queue.put({
            "type": "training_complete",
            "confusion_matrix": cm_list
        }), loop)
        
    except Exception as e:
         asyncio.run_coroutine_threadsafe(async_queue.put({"type": "error", "message": str(e)}), loop)
