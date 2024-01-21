import torch
import itertools
import numpy as np
import torch.nn as nn
from typing import List
from numpy import ndarray
import matplotlib.pyplot as plt
from time import process_time_ns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from datetime import datetime
from tqdm import tqdm


class Model(nn.Module):
    """
    Clase con la inciciañización de i red neuronal, función que hace de propagación hacia delante
    Función de entrenamiento, función de evaluación
    Tiene como argumentos de entrada: las entradas de la red, el numero de neuronas de la capa oculta y
    el numero de clases a clasificar
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # Aplica una transformación lineal a los datos entrantes
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x) -> torch.Tensor:
        """
        Propagación hacia delante
        devuelve la salida de la red neuronal
        """
        out1 = self.fc1(x)
        out2 = self.relu(out1)
        out3 = self.fc2(out2)
        return out3
def fit(model_to_fit, num_epochs, train_loader, learning_rate, test_loader) -> List[int] and List[float] and List[float]\
                                                                           and List[float]:
    """
    Función de entrenamiento, un entrenamiento por epocas y lotes en cada una de ellas
    Argumentos de entrada son: el propio modelo, numero de epocas,
    el conjunto de datos de entrenamiento,y el learning rate
    Devuelve: una lista con la perdida y el numero de iteraciones(muestras)
    """
    # Función de perdida
    criterion = torch.nn.CrossEntropyLoss()

    #  para la etapa de actualización de pesos
    optimizer = torch.optim.Adam(model_to_fit.parameters(), lr = learning_rate)
    # optimizer = torch.optim.SGD(self.parameters(), lr=0.02)
    # optimizer = torch.optim.Adagrad(self.parameters())
    # optimizer = torch.optim.RMSprop(self.parameters())
    # optimizer = torch.optim.Adadelta(self.parameters())
    # optimizer = torch.optim.Adam(self.parameters())


    # inicializamos el modelo con los parametros de entrada de la función de perdida y el optimizador
    train_func = train(model_to_fit, optimizer, criterion)

    total = 0
    loss_list, iters = [], []
    list_acc_train_aux,  list_acc_test_aux = [], []
    list_acc_train, list_acc_test = [], []
    error_test_old, error_test_new = 0,0
    loss_test_list_epoc, loss_test_list = [], []

    patience = 0
    overfitting = False

    bar = tqdm(total = num_epochs, position =0, leave=False)

    for e in range(1,num_epochs + 1):
        loss_list_epoc = []
        iters.append(total)
        if loss_test_list_epoc:
            error_test_old = np.mean(loss_test_list_epoc)

        for (X, y), (X_test, y_test) in zip(train_loader, test_loader):
            loss, modelo_train = train_func(X.float(), y)
            loss_list_epoc.append(loss)
            total += len(y)
                
            out_test = modelo_train(X_test.float())
            loss_test = criterion(out_test, y_test)
            loss_test_list_epoc.append(loss_test.item())

            accuracy_train = evaluate(model_to_fit,train_loader)
            list_acc_train_aux.append(accuracy_train)

            accuracy_test = evaluate(model_to_fit,test_loader)
            list_acc_test_aux.append(accuracy_test)

        bar.set_description(f'loss {np.mean(loss_list_epoc):.5f} acc {np.mean(list_acc_test_aux):.5f}')
        bar.update(1)

        error_test_new = np.mean(loss_test_list_epoc)

        list_acc_train.append(np.mean(list_acc_train_aux))
        list_acc_test.append(np.mean(list_acc_test_aux))
        loss_list.append(np.mean(loss_list_epoc))
        loss_test_list.append(np.mean(loss_test_list_epoc))

        if e % 10 == 0:
            overfitting = early_stopping(loss_test_list)
            if overfitting and patience == 2:
                """ list_acc_train.append(np.mean(list_acc_train_aux))
                list_acc_test.append(np.mean(list_acc_test_aux))
                loss_list.append(np.mean(loss_list_epoc))
                loss_test_list.append(np.mean(loss_test_list_epoc)) """
                print(f'OVERFFITING|Epochs: {e}/{num_epochs}|Loss test old: {error_test_old}|loss test new: {error_test_new}')
                break
            elif overfitting and patience < 2:
                patience += 1
            else:
                continue
    return iters, loss_list, list_acc_train, list_acc_test, loss_test_list, e

def evaluate(model_to_evaluate, test_loader) -> ndarray:
    """
    Evalua el modelo previamente entrnado
    Argumentos de entrada: modelo, conjunto de test
    """
    model_to_evaluate.eval()
    acc = []
    with torch.no_grad():
        for batch in test_loader:
            X, y = batch
            y_hat = model_to_evaluate(X.float())
            acc.append((y == torch.argmax(y_hat, axis=1)).sum().item() / len(y))
    return np.mean(acc)

def train(modelo, optimizer, loss_fn) -> float:
    """
    trainig loop
    Proceso de forward y back (actualización de pesos)
    Argumentos de entrada: modelo, función de perdida y optimizador
    """
    def train_step(x, y) -> float:
        modelo.train()
        optimizer.zero_grad()
        out = modelo(x)
        #print(f'Tamaño de la predicción: {out.size()}')
        #print(f'Tamaño de lo verdadero: {y}')
        loss = loss_fn(out, y)
        #print(f'Tamaño de la perdida: {loss.shape}')
        loss.backward()
        optimizer.step()
        modelo_train = modelo
        return loss.item(), modelo_train
    return train_step

def early_stopping(loss_test_list):
    trend_validation = [b - a for a, b in zip(loss_test_list[::1], loss_test_list[1::1])]
    pibote = int(len(trend_validation) - 10)
    watch_trend = trend_validation[pibote:]
    overfitting = True
    cont = 0
    while overfitting and cont < len(watch_trend):
        if watch_trend[cont] < 0:
            overfitting = False
        cont += 1
    return overfitting

def get_score(modelo_to_fit, train_loader, test_loader, num_epochs, learning_rate) -> List[int] and List[float] and float \
                                                                               and float:
    """
    Función que entrena, evalua y obtine tiempo de entrnamiento de un modelo
    Argumentos de entrada son el modelo, datos de train, datos de test, numero de epocas, learning rate
    Salida: iteraciones, lista de loss, accuracy y tiempo de entrnamiento
    """
    iters, loss_list, acc_train, acc_test, loss_test_list, epochs = fit(modelo_to_fit,num_epochs, train_loader, learning_rate, test_loader)
    accuracy = evaluate(modelo_to_fit,test_loader)
    return iters, loss_list, accuracy, acc_train, acc_test, loss_test_list, epochs


def CV_Kfold(device, num_classes, input_size, X_ten, y_ten, hidden_size, num_splits, batch_size, num_epochs,
             learning_rate, Test, modelo_pre_train, class_names, labels) -> List[int] and List[float] and List[float] and Model and float:
    """
    Función que implementa la validación cruzada con k-Fold
    Argumentos de entrada: dispositivo(cpu, gpu), numero de clases, numero de entradas de la red, datos de entrnamiento
    y test separados por variable clasificatoria y el resto, numero de neuronas capa oculta, la k de k-Fold,
    numero de epocas, tasa de aprendizaje
    Devuelve: iterador y loss para la grafica, lista de accuracys del k-fold, modelo  y tiempo medio de entrenamiento
    """
    folds = StratifiedKFold(n_splits=num_splits, shuffle=True)

    score_mlp = []
    cont = 1

    t1_start = process_time_ns()
    acc_model_to_save = 0
    model_to_return = None
    list_acc_test_return = []
    for train_index, test_index in folds.split(X_ten, y_ten):
        X_tensor = torch.tensor(X_ten).cuda()
        y_tensor = torch.tensor(y_ten).cuda()
        """ X_tensor = torch.tensor(X_ten)
        y_tensor = torch.tensor(y_ten)
 """
        if modelo_pre_train:
            print(f'Entro en modelo preentrnado')
            model = modelo_pre_train.to(device)
        else:
            model = Model(input_size, hidden_size, num_classes).to(device)
            #model = Model(input_size, hidden_size, num_classes)

        X_train, X_test, y_train, y_test = X_tensor[train_index], X_tensor[test_index],\
                                           y_tensor[train_index], y_tensor[test_index]

        
        trainset = torch.utils.data.TensorDataset(X_train, y_train)
        testset = torch.utils.data.TensorDataset(X_test, y_test)

        """ train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True) """
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


        iters, loss_list, accuracy, acc_train, acc_test, loss_test_list, epochs = get_score(model, train_loader, test_loader, num_epochs,learning_rate)
        

        score_mlp.append(accuracy)
        
        #Saco metricas
        graficas(epochs, acc_train,acc_test, iters, loss_list, loss_test_list)
        y_true_test, y_pred_test, acc_test = metricas_accuracy(score_mlp, X_ten, y_ten, Test[0], Test[1], model)
        matrices_confusión(y_true_test, y_pred_test, class_names, labels)

        list_acc_test_return.append(acc_test)

        #Scripting save
        now = datetime.now()
        scripted_model = torch.jit.script(model.cpu())
        scripted_model.save('model_sp_'+str(now.date())+'_'+str(round(acc_test, 4))+str(cont)+'.zip')
        cont += 1

        if acc_test >= acc_model_to_save:
            #Scripting save
            scripted_model_final = torch.jit.script(model.cpu())
            acc_model_to_save = acc_test

    scripted_model_final.save('modelo_final_'+str(now.date())+'_'+str(round(acc_test, 4))+'_Train'+'.zip')
    model_to_return = model.to(device)

    t1_stop = process_time_ns()
    time = (t1_stop - t1_start) / 1000000000
    # Muestro el tiempo medio del entrenamiento de mi red neuronal 
    if(time> 3600):
        print(f'Time: {int(time/3600)}h {int((time%3600)/60)}m {round((time%3600)%60)}s')
    elif(time> 60):
        print(f'Time: {int(time /60)}m {time%60}s')
    else:
        print(f'Time: {time}')
    #return iters, loss_list, score_mlp, modelo, time, acc_train, acc_test
    return time, model_to_return, loss_test_list, np.mean(list_acc_test_return)


def plot_confusion_matrix(label, cm, classes, 
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('\n ')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel(label[0])
    plt.xlabel(label[1])
    plt.tight_layout()

def graficas(num_epochs, accu_train,accu_test, iters, loss_list, loss_test_list):
    epochs = [i for i in range(num_epochs)]
    #Grafica de la función de perdida
    plt.plot(epochs,accu_train,'r' , epochs,accu_test,'g', label ='Accuracy / epochs')
    plt.legend(['Accuracy train', 'Accuracy test'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('>>>>>Accuracy>>>>>')
    plt.show()


    # #Grafica de la función de perdida
    plt.plot(iters, loss_list, "b", iters, loss_test_list,"y", label = 'Loss train / Loss test')
    plt.legend(['Loss Train', 'Loss Test'])
    plt.xlabel('Samples')
    plt.ylabel('Loss')
    plt.title('>>>>>Loss train and test>>>>>')
    plt.show()

    # #Grafica de la función de perdida
    #plt.plot(iters, loss_test_list, color="y")
    #plt.legend(['Loss_test'])
    #plt.xlabel('Samples')
    #plt.ylabel('Loss')
    #plt.title('>>>>>Loss Test>>>>>')
    #plt.show()

def metricas_accuracy(score_mlp, X_ten, y_ten, X_test, y_test, modelo):
    # Media de accuracy obtenidos en el k-Fold
    np.mean(score_mlp)

    #Media de accuracy obtenidos en el k-Fold
    media_accuracy_kFold = np.mean(score_mlp)
    print(f'Accuracy de validación con kFold {media_accuracy_kFold}')

    #Para copiar la construcción de un tensor, se recomienda utilizar
    #sourceTensor.clone().detach() o sourceTensor.clone().detach().requires_grad_(True), 
    #en lugar de torch.tensor(sourceTensor). X_trn = torch.tensor(X_train).cuda()
    # Calculo el accuracy con el conjunto que se ha usado para entrenamiento (el que se le pasa al kFold)
    X_trn = torch.tensor(X_ten).cuda()
    y_trn = torch.tensor(y_ten).cuda()
    """ X_trn = torch.tensor(X_ten)
    y_trn = torch.tensor(y_ten) """

    out_train = modelo(X_trn.float())

    y_pred_train = torch.max(out_train,1)[1].cpu().detach().numpy()
    y_true_train = y_trn.cpu().detach().numpy()

    acc_train = accuracy_score(y_true_train, y_pred_train)
    print('\033[5;31m'+'Accuracy Train:', acc_train,'\033[0;m')

    #Calculo el accuracy con el conjunto de test (núnca visto por el modelo antes)
    X_tst = torch.tensor(X_test).cuda()
    y_tst = torch.tensor(y_test).cuda()
    """ X_tst = torch.tensor(X_ten)
    y_tst = torch.tensor(y_ten) """

    out_test = modelo(X_tst.float())

    y_pred_test = torch.max(out_test,1)[1].cpu().detach().numpy()
    y_true_test = y_tst.cpu().detach().numpy()

    acc_test = accuracy_score(y_true_test, y_pred_test)
    print('\033[5;32m'+'Accuracy Test:',acc_test,'\033[0;m')

    #Diferencia entre accuracy de train y el de test
    print(f'Diferencia entre train y test: {acc_train - acc_test}')

    return y_true_test, y_pred_test, acc_test

def matrices_confusión(y_true_test, y_pred_test, class_names, labels):
    #Inicializamos la matriz de confusión con los datos de test predichos;   Eje y: clase real     Eje x: clase predicha
    #mx_confusion=confusion_matrix(y_true_test, y_pred_test, labels = [1,2,3,4,5,6,7,8,9,10,11,12])
    #mx_confusion=confusion_matrix(y_true_test, y_pred_test, labels = [1,2,3,4,5,6,7,8,10,11])
    mx_confusion=confusion_matrix(y_true_test, y_pred_test, labels = labels)
        
    #Graficamos la matriz de confusión
    # class_names = [1,2,3,4,5,6,7,8,9,10,11,12]
    #class_names = ['Walking', 'Running', 'Going up', 'Going down', 'Sitting', 'Sitting down', 'Standing up', 'Standing', 
    #                'Bicycling', 'Up by elevator',  'Down by elevator',  'Sitting in car', ] 
    """ class_names = ['Walking', 'Running', 'Going up', 'Going down', 'Sitting', 'Sitting down', 'Standing up', 'Standing',
                  'Up by elevator',  'Down by elevator' ] """ 
    label = ['True label','Predicted label']
    plt.figure(figsize=(12,8))
    
    plot_confusion_matrix(label,mx_confusion, classes=class_names, title='Confusion matrix, without normalization')
    plt.show()

    ############## DEPRECATED ###################
    #  FP = mx_confusion.sum(axis=0) - np.diag(mx_confusion)  
    #     FN = mx_confusion.sum(axis=1) - np.diag(mx_confusion)
    #     TP = np.diag( mx_confusion)
    #     TN = mx_confusion.sum() - (FP+FN+TP)
    #     FP = FP.sum()
    #     FN = FN.sum()
    #     TP = TP.sum()
    #     TN = TN.sum()
    #     matrix = np.array([TP,FN,FP, TN]).reshape((2,2))
    #     label = ['Predicted label','True label']
    #     plot_confusion_matrix(label,matrix, classes=['positive', 'negative'], 
    #                                     title='Confusion matrix')
    #     plt.show()
    #     # Sensitivity, hit rate, recall, or true positive rate
    #     TPR = TP/(TP+FN)
    #     print(f'RECALL >>>>>> {TPR}')
    #     # Precision or positive predictive value
    #     PPV = TP/(TP+FP)
    #     print(f'PRECISION >>>>>> {PPV}')
    #     # Negative predictive value
    #     NPV = TN/(TN+FN)
    #     print(f'ESPECIFICITY >>>>>> {NPV}')
    #     # Overall accuracy
    #     ACC = (TP+TN)/(TP+FP+FN+TN)
    #     print(f'ACCURACY >>>>>> {ACC}\n')

    # # Fall out or false positive rate
    # FPR = FP/(FP+TN)
    # # False negative rate
    # FNR = FN/(TP+FN)
    # # False discovery rate
    # FDR = FP/(TP+FP)
    # # Specificity or true negative rate
    # TNR = TN/(TN+FP) 
