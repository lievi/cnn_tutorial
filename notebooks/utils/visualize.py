import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_accuracy_and_loss(history):

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(
        history.history['loss'],
        color='b',
        label='Training Loss'
    )
    ax[0].plot(
        history.history['val_loss'],
        color='r',
        label='Validation Loss'
    )
    legend = ax[0].legend(loc='best')

    ax[1].plot(
        history.history['accuracy'],
        color='b',
        label="Training Accuracy"
    )
    ax[1].plot(
        history.history['val_accuracy'],
        color='r',
        label="Validation Accuracy"
    )
    legend = ax[1].legend(loc='best')

    print(
        f"Best accuracy: {max(history.history['accuracy']):.6f}"
    )
    print(
        "Best validation accuracy"
        f": {max(history.history['val_accuracy']):.6f}"
    )