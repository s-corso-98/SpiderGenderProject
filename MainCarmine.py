import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.svm import SVC
import time

start_time = time.time()

#Caricamento dei due dataset
dataset = pd.read_csv("DatasetCelebA/dataset4c3s.csv",header=None)
feature = pd.read_csv("DatasetCelebA/list_attr_celeba.csv")

#Imposto il numero di righe
num_righe=202599

#Salvo in datafram le prime 100 righe di dataset
dataframe = dataset.iloc[0:num_righe, 0:48]

#salvo in feat le prime 100 righe
feat = feature.iloc[0:num_righe,21]
df_X = pd.DataFrame(feat)

#rinonimo la colonna da Male a Gender
rename = df_X.rename(columns={"Male" : "Gender"}) #-1 donna e 1 maschio

#Concateno i due dataframe per crearne uno
dfconc = pd.concat([dataframe, rename], axis=1, sort=False)
print(dfconc)

#Salvo in X_split le prime 50 righe e 64 colonne per i landmark
X_split = dfconc.iloc[0:num_righe,0:48].values
#Salvo in Y_split le prime 50 righe e la colonna Gender
Y_split = dfconc.iloc[0:num_righe, 48].values #classi

# Splitting del dataset in Training set e Test set rispettivamente 70% e 30%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_split, Y_split, test_size = 0.3, random_state = 0)

#Stampo del numero di elementi di Training
print(X_train.shape)
print(y_train.shape)

#Stampa del numero di elementi di Test
print(X_test.shape)
print(y_test.shape)

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
plt.savefig('ConfusionMatrix.png', bbox_inches='tight')
plt.show()

#Stampa dell'accuratezza
from sklearn.metrics import accuracy_score
print("Accuratezza: ")
print(accuracy_score(y_test, predicted))


