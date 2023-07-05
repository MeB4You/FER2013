import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from ResNet import ResNet18
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import load_model
# from keras.optimizers import Adam


def get_model(num_labels):
    return ResNet18(num_labels,(48,48,1))


def plot_acc_loss(history,e_range):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(e_range)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def show_model_architectures(model):
    model.summary()

def compile_model(model):
    model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
    optimizer= tf.keras.optimizers.Adam(learning_rate = 1e-4), 
    metrics=['accuracy']
    )

def get_callbacks():
    mcp_save = ModelCheckpoint(filepath = 'models/Resnet18', 
        save_freq = 'epoch' ,
        save_best_only = True,
        monitor = 'val_loss',
        mode ='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1, mode='min',min_lr= 1e-5)

    callbacks = [mcp_save,reduce_lr_loss]
    return callbacks

def load_best_model():
    return load_model('models/Resnet18')

def plot_confusion_matrix(y_prediction,y_test):
    labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    cm = confusion_matrix(y_test, y_prediction, normalize= 'true')
    disp = ConfusionMatrixDisplay(cm, display_labels = labels)
    disp.plot(xticks_rotation = 'vertical',cmap=plt.cm.Blues)
    plt.show()