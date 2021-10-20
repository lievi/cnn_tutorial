import pandas as pd
import numpy as np

np.random.seed(2)

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def normalize_data(X_train, test):
    # Transforming the data, that is of range [0..255],
    # to [0..1]
    X_train = X_train / 255.0
    test = test / 255.0
    return X_train, test


def reshape_data(X_train, test):
    """
    Reshape the images in 3 dimensions
    28px height, 28px width and 1 color channel
    """
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)
    return X_train, test


def split_train_and_validation_data(X_train, Y_train):
    random_seed = 2
    return train_test_split(
        X_train, Y_train, test_size=0.1, random_state=random_seed
    )


train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# getting the label (the "response") from the train data
Y_train = train['label']

# label encoding
Y_train = to_categorical(Y_train, num_classes=10)

# removing the label from the train data
X_train = train.drop(labels=["label"], axis=1)

X_train, test = normalize_data(X_train, test)

X_train, test = reshape_data(X_train, test)

X_train, X_val, Y_train, Y_val = split_train_and_validation_data(
    X_train, Y_train
)

np.savez(
    'data/processed/processed_data',
    X_train=X_train,
    X_val=X_val,
    Y_train=Y_train,
    Y_val=Y_val
)
