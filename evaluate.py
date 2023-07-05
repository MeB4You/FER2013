from utils import (load_best_model,plot_confusion_matrix)
from load_images import (get_dataset)
import numpy as np
def evaluate():
    model = load_best_model()
    print("Successfully loaded FER Model!")
    _,_,_,_, X_test, y_test = get_dataset()
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred,axis = 1)
    y_pred_accuracy = np.mean(y_pred==y_test)
    print("The accuracy on test data is: ", y_pred_accuracy)
    plot_confusion_matrix(y_pred,y_test)
    
evaluate()