import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import pydotplus
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def MySupportVectorMachine():
   #Creazione del classificatore
   classifier = SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=1, shrinking=True, tol=0.001,
      verbose=False)

   #Si allena il classificatore
   classifier.fit(X_train, y_train)

   # E ora si predice sul Test Set
   predicted = classifier.predict(X_test)

   #Rinomino i campi per la matrice di confusione
   labels = ("Female","Male")
   positions = (0,1)

   #Stampa dei risultati
   print("Classification report for classifier %s:\n%s\n"
         % (classifier, metrics.classification_report(y_test, predicted)))
   disp = metrics.plot_confusion_matrix(classifier, X_test, y_test, cmap="OrRd")
   disp.figure_.suptitle("Confusion Matrix")
   print("Confusion matrix:\n%s" % disp.confusion_matrix)

   #Stampa del tempo intercorso per processare il classificatore
   print ("\nTempo trascorso: {:.2f}m\n".format((time.time()-start_time)/60))

   #Stampa a video della matrice di confusione
   plt.xticks(positions,labels)
   plt.yticks(positions,labels)
   plt.savefig('OutputSVM/ConfusionMatrix.png', bbox_inches='tight')
   plt.show()

   #Stampa dell'accuratezza
   from sklearn.metrics import accuracy_score
   print("Accuratezza: ")
   print(accuracy_score(y_test, predicted))







def MyDecisionTree():
    #The decision tree classifier
    clf = tree.DecisionTreeClassifier(criterion = "gini", max_depth= 13)

    #Alleno il decision tree
    clf_train = clf.fit(X_train, y_train)

    #Predico la risposta per il dataset
    y_pred = clf.predict(X_test)

    #Model Accuracy, valuto il modello
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    #Creo un decision tree in formato DOT utilizzando GraphViz
    dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=X_train.columns.values,
                                    class_names=['Female', 'Male'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
    #Creo il decision tree in formato Graph partendo dal formato DOT
    graph = pydotplus.graph_from_dot_data(dot_data)
    #Salvo in png il decision tree creato
    test2 = graph.write_png("OutputDT/GraphDecisionTree.png")
    print ("\nTempo trascorso: {:.2f}m\n".format((time.time()-start_time)/60))




def MyNearestNeighbors():
    #NearestNeighbors classifier
    classifier = KNeighborsClassifier(n_neighbors=9)

    #Alleno il classificatore
    clf_train = classifier.fit(X_train, y_train)

    #Predico la risposta per il dataset
    y_pred = classifier.predict(X_test)

    labels = ("Female","Male")
    positions = (0,1)

    #Stampa dei risultati
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, y_pred)))
    disp = metrics.plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)

    # Model Accuracy, valuto il modello
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("\nTempo trascorso: {:.2f}m\n".format((time.time() - start_time) / 60))

    #Stampa a video
    plt.xticks(positions,labels)
    plt.yticks(positions,labels)
    plt.savefig('OutputKNN/ConfusionMatrixNoWeb_n9.png', bbox_inches='tight')
    plt.show()




#Funzione creazione dataset bilanciato
def CreateBalanced4c4s(dfconc):
    #Creo un csv corrispondente al dataset4c4s ma con l'aggiunta della colonna "Gender"
    dfconc.to_csv("DatasetCelebA/DatasetBalancedNoWeb.csv", header = False, index = False)
    #Leggo il csv appena creato per andare ad eseguire le operazioni di manipolazione
    DFbalanced = pd.read_csv("DatasetCelebA/DatasetBalancedNoWeb.csv",header = None)

    #Salvo in un dataframe tutte le righe con gender pari a 1(uomo)
    dfBalanceM = DFbalanced.loc[DFbalanced[128] == 1]
    #Salvo in un dataframe tutte le righe con gender pari a -1(donna)
    dfBalanceF = DFbalanced.loc[DFbalanced[128] == -1]

    #Droppo le righe in eccesso del dataframe femminile (rispetto al dataframe maschile)
    dfBalanceF = dfBalanceF.iloc[0:432]

    #Unisco i due dataframe aventi lo stesso numero di elementi
    DFbalanced = pd.concat([dfBalanceM,dfBalanceF], axis = 0)

    #Creo il csv corrispondente
    DFbalanced.to_csv("DatasetCelebA/DatasetBalancedNoWeb.csv", header = False, index = False)




#Funzione per suddividere il dataset bilanciato in train e test set
def ExecOnBalanced():
    #Leggo dataset bilanciato
    dataframe = pd.read_csv("DatasetCelebA/DatasetBalancedNoWeb.csv", header=None)

    #Rinomino la colonna 64 in Gender.
    dataframe = dataframe.rename(columns={dataframe.columns[128]: "Gender"})  # -1 donna e 1 maschio

    #Ottengo feature variables
    feature_cols = list(dataframe.columns.values)
    X = feature_cols[1:len(feature_cols) - 1]
    X = dataframe[X]

    #Ottengo target variables
    y = dataframe.Gender

    #Divido il dataframe in train e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test
    return X_train,X_test,y_train,y_test






#Funzione lettura CSV su cui eseguire i test
def ReadCSV():
    # Caricamento dei due dataset
    dataframe = pd.read_csv("DatasetCelebA/datasetnoweb.csv", header=None)
    feature = pd.read_csv("DatasetCelebA/list_attr_celeba.csv")

    # Prendo la colonna delle features riguardante il sesso.
    feat = feature.iloc[0:1003, 21]
    df_X = pd.DataFrame(feat)

    # Rinonimo la colonna da Male a Gender.
    rename = df_X.rename(columns={"Male": "Gender"})  # -1 donna e 1 maschio

    # Concateno i due dataframe per crearne uno. Non necessario per il dataset bilanciato
    dfconc = pd.concat([dataframe, rename], axis=1, sort=False)
    return dfconc





#Ottengo l'orario corrente per andare a calcolare il tempo di esecuzione totale di ciascun classificatore
start_time = time.time()

choice = input("Digita 1 se vuoi eseguire i test sul dataset bilanciato, 0 altrimenti.\n")
if choice == "1":
    #Eseguo la funzione per leggere il dataset di interesse per i test. In questo caso serve a creare il dataset bilanciato
    dfconc = ReadCSV()
    #Eseguo la funziona per creare il dataset bilanciato 4c4s
    CreateBalanced4c4s(dfconc)
    #Salvo nelle variabili il dataframe diviso in train e test set. In modo da potere poi passare ai classificatori
    X_train, X_test, y_train, y_test = ExecOnBalanced()
else:
    #Eseguo la funzione per leggere il dataset di interesse per i test
    dfconc = ReadCSV()

    #Ottengo feature variables
    feature_cols = list(dfconc.columns.values)
    X = feature_cols[1:len(feature_cols)-1]
    X = dfconc[X]

    #Ottengo target variables
    y = dfconc.Gender

    #Divido il dataframe in train e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#Esecuzione classificatori
#MyDecisionTree()
#MyNearestNeighbors()
MySupportVectorMachine()