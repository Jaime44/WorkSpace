from time import process_time_ns
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import validation
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import matplotlib.axes as axs
from sklearn.metrics import plot_confusion_matrix
import joblib
from datetime import datetime

#Modelo SVM sin PCA
def modl(model,Df_X, Df_y, num_splits, model_name, name_class):
    
    folds = StratifiedKFold(n_splits=num_splits)

    scores_svm_train, scores_svm_test = [], []

    model_to_return = None
    acc_model_to_save = 0

    t1_start = process_time_ns()
    k = 1

    for train_index, test_index in folds.split(Df_X.values,Df_y.values):
        
        X_train, X_test, y_train, y_test = Df_X.values[train_index], Df_X.values[test_index], \
                                           Df_y.values[train_index], Df_y.values[test_index]

        mod_fit = model.fit(X_train, y_train)

        #acc_train = mod_fit.score(X_train, y_train)
        #scores_svm_train.append(acc_train)
 
        acc_test = mod_fit.score(X_test, y_test)
        scores_svm_test.append(acc_test)

        print(f'Accuracy test en split {k} --> {acc_test}')
        confusion_matrix(mod_fit, X_test, y_test, name_class)
        k += 1
        if acc_test >= acc_model_to_save:
            acc_model_to_save = acc_test
            model_to_return = mod_fit
        
    t1_stop = process_time_ns()
    time = (t1_stop-t1_start)/1000000000

    # Muestro el tiempo medio del entrenamiento de mi red neuronal 
    get_time(time)
    confusion_matrix(model_to_return, X_test, y_test, name_class)

    now = datetime.now()
    # save the model to disk
    filename = 'modelo_final_'+str(now.date())+'_'+str(round(acc_test, 4))+'_Train_'+str(model_name)+'.sav'
    joblib.dump(model_to_return, filename)
 
    # some time later...
 
    # load the model from disk
    # loaded_model = joblib.load(filename)
    # result = loaded_model.score(X_test, Y_test)
    # print(result)

    #print(f'\033[5;31mAccuracy medio de train --> {np.mean(scores_svm_train)}\033[0;m')
    return scores_svm_test, (t1_stop-t1_start)/1000000000, model_to_return

""" def plot_graphic_cross_validation(accu_train, accu_test):
    epochs = 1000 * np.arange(1,11)
    #Grafica de la función de perdida
    plt.plot(epochs,accu_train,'r' , epochs,accu_test,'g', label ='Accuracy / epochs')
    plt.legend(['Accuracy train', 'Accuracy test'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('>>>>>Accuracy>>>>>')
    plt.show() """

def get_time(time_mean):
    if(time_mean> 3600):
        print(f'Time: {int(time_mean/3600)}h {int((time_mean%3600)/60)}m {round((time_mean%3600)%60)}s')
    elif(time_mean> 60):
        print(f'Time: {int(time_mean /60)}m {time_mean%60}s')
    else:
        print(f'Time: {time_mean}s')


def confusion_matrix(classifier, X_test, y_test, name_class):
    
    """ class_names = ['Walking', 'Running', 'Going_U', 'Going_D', 'Sitting', 'Sitting_D', 'Standing_U', 'Standing',
                  'U_elevator',  'D_elevator' ] """
    
    class_names = name_class
    fig, ax = plt.subplots(figsize=(20, 10))
    disp = plot_confusion_matrix(classifier, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues, ax = ax, colorbar=False)
    disp.ax_.set_title('Confusion matrix')
    #print(disp.confusion_matrix)
    plt.show()