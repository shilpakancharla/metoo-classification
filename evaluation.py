import matplotlib.pyplot as plt
from keras import backend as K

def max_seq_length(sequence):
    length = []
    for i in range(0, len(sequence)):
        length.append(len(sequence[i]))
    return max(length)

def recall_measure(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_measure(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision_measure(y_true, y_pred)
    recall = recall_measure(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def plot_history(history):
    # Plot loss
    plt.title('Loss')
    plt.plot(history.history['loss'], color = 'blue', label = 'train')
    plt.plot(history.history['val_loss'], color = 'red', label = 'test')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

    # Plot accuracy
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color = 'blue', label = 'train')
    plt.plot(history.history['val_accuracy'], color = 'red', label = 'test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

    # Plot F1
    plt.title('F1-Score')
    plt.plot(history.history['f1'], color = 'blue', label = 'train')
    plt.plot(history.history['val_f1'], color = 'red', label = 'test')
    plt.ylabel('F1-Score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

    # Plot precision
    plt.title('Precision')
    plt.plot(history.history['precision_measure'], color = 'blue', label = 'train')
    plt.plot(history.history['val_precision_measure'], color = 'red', label = 'test')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

    # Plot recall
    plt.title('Recall')
    plt.plot(history.history['recall_measure'], color = 'blue', label = 'train')
    plt.plot(history.history['val_recall_measure'], color = 'red', label = 'test')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()