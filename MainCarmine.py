import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


#Prova generale con 20 foto, lo si estende a tutto il dataset una volta
#raffinato il codice

#caricamento dei due dataset
dataset = pd.read_csv("DatasetCelebA/dataset4c4s.csv",header=None)#modificare nome dataset in base alla configurazione scelta
feature = pd.read_csv("DatasetCelebA/list_attr_celeba.csv")

#print (dataset.shape)
#print (type(dataset))

#Salvo in datafram le prime 20 righe di dataset
dataframe = dataset.iloc[0:20, 0:64]

#print(feature.shape)
#print(type(feature))

#salvo in feat le prime 20 righe
feat = feature.iloc[0:20,21]
df_X = pd.DataFrame(feat)
#rinonimo la colonna da Male a Gender
rename = df_X.rename(columns={"Male" : "Gender"}) #-1 donna e 1 maschio
#print(rename)

#Concateno i due dataframe per crearne uno
dfconc = pd.concat([dataframe, rename], axis=1, sort=False)
print(dfconc)
#Stampa intero dataset
#print (dataset)

#print (list(feature))

#colonna del gender per le prime 8 foto
#feat = feature.iloc[0:2,[0,21]]


#merge
#result = pd.merge(dataset,rename,left_on=' ',right_on='Gender',how='left')
#print(result)

#Salvo in X_split le prime venti righe e 64 colonne per i landmark
X_split = dfconc.iloc[0:20,0:64].values
#Salvo in Y_split le prime venti righe e la colonna Gender
Y_split = dfconc.iloc[0:20, 64].values

# Splitting del dataset in Training set e Test set rispettivamente 80% e 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_split, Y_split, test_size = 0.2, random_state = 0)

print(X_train.shape)
#print(X_train)
print(y_train.shape)
#print(y_train)
print(X_test.shape)
#print(X_test)
print(y_test.shape)
#print(y_test)


#Creazione del classificatore, presi dal tutorial anche gli altri in caso di altre prove

#classifier = svm.SVC(gamma=0.001)
#classifier = KNeighborsClassifier(3)

#classifier = SVC(kernel="linear", C=10)
classifier = SVC(kernel='rbf', random_state=0, gamma=.01, C=1000) #al variare di gamma e C

#classifier = SVC(gamma=2, C=1)
#classifier = GaussianProcessClassifier(1.0 * RBF(1.0))
#classifier = DecisionTreeClassifier(max_depth=5)
#classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#classifier = MLPClassifier(alpha=1, max_iter=1000)
#classifier = AdaBoostClassifier()
#classifier = GaussianNB()
#classifier = QuadraticDiscriminantAnalysis()


#Si allena il classificatore
classifier.fit(X_train, y_train)

# E ora si predice sul Test Set
predicted = classifier.predict(X_test)

print(predicted.shape)
print(y_test.shape)

labels = ("Female","Male")
positions = (0,1)

#Stampa dei risultati
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
#disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)
#Stampa a video
plt.xticks(positions,labels)
plt.yticks(positions,labels)
plt.title("Confusion Matrix")
plt.show()

# Accuratezza
from sklearn.metrics import accuracy_score
print("Accuratezza: ")
print(accuracy_score(y_test, predicted))