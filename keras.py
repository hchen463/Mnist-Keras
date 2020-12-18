
"""
In this file we use Keras to train mnist.

@author: Hongxu Chen
"""

import tensorflow as tf
from tensorflow import keras







def get_dataset(training = True):
    
    """
    Parameters
    ----------
    training : TYPE, optional
        an optional boolean argument (default value is True for training dataset)

    Returns : two NumPy arrays for the train_images and train_labels
    -------

    """
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    if training:
        return (train_images, train_labels)
    else:
        return (test_images, test_labels)
    


def print_stats(train_images, train_labels):
    """
    Parameters
    ----------
    Input: the dataset and labels produced by the previous function

    Returns: None
    -------

    """
    print(len(train_images))
    print(str(len(train_images[0]))+ 'x' + str(len(train_images[0][0])))
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 
                   'Six', 'Seven', 'Eight', 'Nine']
    label = {}
    for i in range(len(train_labels)):
        label[train_labels[i]] = label.setdefault(train_labels[i], 0) + 1
    
    for i in range(len(class_names)):
        print(str(i) + '. ' + class_names[i] + ' - ' + str(label[i]))
    
    
    
(train_images, train_labels) = get_dataset()
# print_stats(train_images, train_labels)

    
    
def build_model():
    """
    takes no arguments and returns an untrained neural network model

    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(28,28)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10))
    opt = keras.optimizers.SGD(learning_rate=0.001)
    l = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    model.compile(optimizer = opt, loss = l, metrics = 'accuracy')
    
    return model
    
model = build_model()
# print(model)  
# print(model.loss)
# print(model.optimizer)
    
    
def train_model(model, train_images, train_labels, T):
    """
    takes the model produced by the previous function 
    and the dataset and labels produced by the first function 
    and trains the data for T epochs; does not return anything
    
    """
    model.fit(x = train_images, y = train_labels, epochs = T)
    

train_model(model, train_images, train_labels, 10)
(test_images, test_labels) = get_dataset(False)    
    
    
def evaluate_model(model, test_images, test_labels, show_loss=True):
    """
    takes the trained model produced by the previous function and the test image/labels, 
    and prints the evaluation statistics as described below 
    (displaying the loss metric value if and only if the optional parameter has not been set to False); 
    does not return anything
    
    """
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    loss = round(test_loss, 4)
    accuracy = "{:.2%}".format(test_accuracy)
    if show_loss:
        print('Loss:' + str(loss))
        print('accuracy:' + str(accuracy))
        
    else:
        print('accuracy:' + str(accuracy))
    
evaluate_model(model, test_images, test_labels)   
    
# model.add(keras.layers.Softmax())    

def predict_label(model, test_images, index):
    """
    takes the trained model and test images, 
    and prints the top 3 most likely labels for the image at the given index, 
    along with their probabilities; does not return anything

    """
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 
                   'Six', 'Seven', 'Eight', 'Nine']
    predict = model.predict(test_images)[index]
    pred = []
    for i in range(10):
        add = [predict[i], i]
        pred.append(add)
    dic = {}
    for i in range(10):
        dic[i] = class_names[i]
    pred.sort(key=lambda x:x[0])
    
    
    print(dic[pred[9][1]] + ': ' + str("{:.2%}".format(pred[9][0])))
    print(dic[pred[8][1]] + ': ' + str("{:.2%}".format(pred[8][0])))
    print(dic[pred[7][1]] + ': ' + str("{:.2%}".format(pred[7][0])))
    
# predict_label(model, test_images, 1)














    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    