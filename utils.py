import numpy as np
import pickle


def load(filename):
    """Load MNIST images
    Returns:
      4 matrices : 'training_images', 'training_labels', 'test_images', 'test_labels'
    """
    with open(filename, 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def preproccess_images(x, y, split):
    """Convert to binary format image, extracts only 0 and 1
    classes from the train data set.
    """
    # convert images from range {0,1,...,255} to range {0,1}
    x = (x > 127).astype(np.uint8)
    # get indices of images, which represent 0 and 1
    indices_of_zeros = np.where(y == 0)
    indices_of_ones = np.where(y == 1)
    # amount of 0 and 1 digits in 'x_train'
    number_zeros = len(indices_of_zeros[0])
    number_ones = len(indices_of_ones[0])
    # get 1 and 0 digits from 'x_train'
    zeros = x[indices_of_zeros, :].reshape((-1, 28, 28))
    ones = x[indices_of_ones, :].reshape((-1, 28, 28))
    n_samples = min(number_ones, number_zeros)
    train_numbers = int(split*n_samples)
    zeros_train = zeros[:train_numbers, ...]
    zeros_test = zeros[train_numbers:n_samples, ...]
    ones_train = ones[:train_numbers, ...]
    ones_test = ones[train_numbers:n_samples, ...]
    x_train = np.concatenate([zeros_train, ones_train], axis=0)
    x_test = np.concatenate([zeros_test, ones_test], axis=0)
    y_train = np.concatenate((np.zeros((train_numbers, 1)), np.ones((train_numbers, 1))), axis=0).astype(np.uint8)
    y_test = np.concatenate((np.zeros((n_samples - train_numbers, 1)),
                             np.ones((n_samples - train_numbers, 1))), axis=0).astype(np.uint8)
    return x_train, y_train, x_test, y_test


def generate_aposterior_probabilities(n_samples,n_classes):
    """Generate a matrix with shape (n_samples,n_classes), where
    sum along axis=1 = 1. p(k|xz)
    """
    rand_matrix = np.random.randint(low=0, high=5000,size=(n_samples, n_classes))
    sum_along_row = np.repeat(rand_matrix.sum(axis=1).reshape((-1, 1)), repeats=2, axis=1)
    prob_matrix = rand_matrix/sum_along_row
    return prob_matrix


def aprior_probabilities(probability_histogram):
    """calculate p(k)
    """
    return probability_histogram.mean(axis=0)


def calculate_parameters(x, aposterior_probs):
    """Calculate p_kij
    Args:
        x : a matrix (8982,28,28)
        aposterior_probs : a matrix (8982,2)
    Returns :
    a matrix of probabilities
    """
    znam = aposterior_probs.sum(axis=0)
    znam = np.expand_dims(znam, axis=1)
    znam = np.repeat(znam, repeats=28*28, axis=1).reshape((2, 28, 28))
    probs = np.expand_dims(aposterior_probs.T, axis=2)
    probs = np.repeat(probs, repeats=28*28, axis=2).reshape((2, x.shape[0], 28, 28))
    x = np.expand_dims(x, axis=0)
    x = np.repeat(x, repeats=2, axis=0)
    return np.multiply(x, probs).sum(axis=1)/znam


def calculate_conditional_probs(x, parameters):
    """ p_kij
    x - (8982,28,28)
    parameters - (2,28,28)

    Returns : 
    a matrix of shape (8292,2)
    """
    parameters = np.expand_dims(parameters, axis=0)
    parameters = np.repeat(parameters, repeats=x.shape[0], axis=0)
    parameters = parameters.reshape((parameters.shape[0], parameters.shape[1], -1))
    x = np.expand_dims(x, axis=1)
    x = np.repeat(x, repeats=2, axis=1)
    x = x.reshape((x.shape[0], x.shape[1], -1))
    power1 = np.power(parameters, x).prod(axis=-1)
    power2 = np.power(1 - parameters, (1 - x)).prod(axis=-1)
    return np.multiply(power1, power2)


def calculate_aposterior(conditions, aprior_probs):
    """
    conditions - (8982,2)
    aprior_probs - (2,)
    Returns :
    a matrix with shape (8292,2)
    """
    znam = conditions[:, 0]*aprior_probs[0] + conditions[:, 1]*aprior_probs[1]
    aposterior0 = (conditions[:, 0]*aprior_probs[0])/znam
    aposterior1 = (conditions[:, 1]*aprior_probs[1])/znam
    return np.stack((aposterior0, aposterior1), axis=1)


def evaluate(x, y, params):
    pred = calculate_conditional_probs(x, params).argmax(axis=1)
    error = (pred - y.flatten()).sum()/len(pred)
    if error < 0:
        pred = 1 - pred
        error = (pred - y.flatten()).sum()/len(pred)
    return error


