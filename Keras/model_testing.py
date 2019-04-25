"""
A Testing and analysis Framework for Saved keras Models 
"""
from __future__ import print_function
import keras
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from keras.models import model_from_json, load_model
from keras.utils import plot_model
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
from keras_model_options import Han_model
import matplotlib.pyplot as plt
import utill
import os
import numpy as np
import datetime

def l_model(model_func, model_weights_h5):
    
    model = model_func()    
    model.load_weights(model_weights_h5)
    return model

def load_weights_testcase():
    ## Load Data ##
    cwd = os.getcwd()
    model = Han_model()
    model_weights_h5 = os.path.join(cwd,"\\Models\\model_03032019_0.h5")
    model.load_weights(model_weights_h5)
    return model

def confusion_matix_testcase():
    y = np.load('y_true.npy')
    y = utill.mutilabel2single(y)
    utill.plot_confusion_matrix(y,y,utill.CLASS_NAMES)

def windowed_predict(model, x_test):
    
    
    results = []
    i = 0 
    
    while i+44 < 44*3:
        x_window = x_test[:,i:i+44]
        x_window = x_window.reshape(1,128, 44, 1)
        result = model.predict(x_window)
        results.append(result)
        i += 22
    
    # results = np.asarray(results)
    return results

def normalize_sum(windowed_results, theta=0.5):
    summed = np.sum(windowed_results, axis=0)
    label = summed/summed.max()
    row,col=label.shape
    label = label.reshape(col,)
    # Can write an if statment to detect if theta is a list, and then use a zip in list comprehension
    label[label < theta] = 0 
    return label

def multiclass_PnR(Y_test,y_score):

    n_classes = y_score[0].size
    if Y_test[0].size != n_classes:
        Y_test = [y[0:n_classes] for y in Y_test]
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    Y_test = np.asarray(Y_test)
    y_score = np.asarray(y_score)
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
    average_precision["macro"] = average_precision_score(Y_test, y_score, average="macro")       
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
    # print('Average precision score, macro-averaged over all classes: {0:0.2f}'.format())
    return precision, recall, average_precision

def multiclass_F1(Y_test,y_score, average_type = 'micro'):
    n_classes = y_score[0].size
    if Y_test[0].size != n_classes:
        Y_test = [y[0:n_classes] for y in Y_test]
    Y_test = np.asarray(Y_test)
    y_score = np.asarray(y_score)
    tp = get_truepos(Y_test,y_score)
    fn = get_falseneg(Y_test,y_score)
    fp = get_falsepos(Y_test,y_score)

    if average_type == 'micro' and hasattr(tp,"__len__"):
        precision = tp.sum()/(tp.sum() +fp.sum())
        recall = tp.sum()/(tp.sum() + fn.sum())
        F1 = 2 * (precision * recall)/(precision + recall)
    elif average_type =='macro' and hasattr(tp,"__len__"):
        uniq = np.unique(Y_test,axis = 0)
        uniq = uniq.dot(1<< np.arange(uniq.shape[-1] -1, -1, -1))
        Y_test_int  = Y_test.dot(1<< np.arange(Y_test.shape[-1] -1, -1, -1))
        Pr_list, Rec_list = [],[]
        for u in uniq:
            tp_ = tp[Y_test_int == u]
            fn_ = fn[Y_test_int == u]
            fp_ = fp[Y_test_int == u]
            Pr_list.append(tp_.sum()/(tp_.sum() +fp_.sum()))
            Rec_list.append(tp_.sum()/(tp_.sum() + fn_.sum()))
        precision = np.mean(np.asarray(Pr_list))
        recall = np.mean(np.asarray(Rec_list))
        F1 = 2 * (precision * recall)/(precision + recall)
    else:
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        F1 = 2 * (precision * recall)/(precision + recall)

    return F1, precision, recall

    
    
def get_truepos(Y_test,y_score):
    log_array = np.logical_and(Y_test,y_score)
    if len(log_array.shape) == 2:
        true_pos = log_array.sum(axis = 1)
    else: 
        true_pos = log_array.sum()
    return true_pos

def get_falseneg(Y_test,y_score):
    log_array = np.logical_and(np.logical_not(y_score),Y_test)
    if len(log_array.shape) == 2:
        false_neg = log_array.sum(axis = 1)
    else: 
        false_neg = log_array.sum()
    return false_neg

def get_falsepos(Y_test,y_score):
    log_array = np.logical_and(np.logical_not(Y_test),y_score)
    if len(log_array.shape) == 2:
        false_pos = log_array.sum(axis = 1)
    else: 
        false_pos = log_array.sum()
    return false_pos

def test():
    gt = np.array([0,0,1,1])
    p1 = np.array([0,0,1,0])
    p2 = np.array([0,1,1,0])
    p3 = np.array([1,1,1,1])
    for p in [gt,p1,p2,p3]:
        f1,precision,recall = multiclass_F1(gt,p)
        print("F1: {0:0.2f}  Pr: {1:0.2f}  Re: {2:0.2f}".format(f1,precision,recall))

if __name__ == "__main__":
    cwd = os.getcwd()
    model_weights = os.path.join(cwd, 'Keras\\trained_models\\Han_recreate_04-11-2019-01_weights.h5 ')
    model = l_model(Han_model,model_weights)

    X_test, y_test = utill.read_test_npz_folder('Keras\\IRMAS_testdata\\')

    start_scoreing = datetime.datetime.now()
    y_score = [normalize_sum(windowed_predict(model,X)) for X in X_test]
    end_scoreing = datetime.datetime.now() -start_scoreing
    print(end_scoreing)

    F1, precision, recall = multiclass_F1(y_test,y_score)
    print("Micro Scores: \n\tF1: {0:0.2f}  Pr: {1:0.2f}  Re: {2:0.2f}".format(F1,precision,recall))
    F1, precision, recall = multiclass_F1(y_test,y_score, average_type='macro')
    print("Macro Scores: \n\tF1: {0:0.2f}  Pr: {1:0.2f}  Re: {2:0.2f}".format(F1,precision,recall))

    
