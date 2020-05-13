import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL.Image import Image
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_learning_curves
from sklearn import svm, metrics
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, validation_curve, ShuffleSplit, learning_curve, StratifiedKFold
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
import time

start_time = time.time()

#caricamento dei due dataset
dataset = pd.read_csv("DatasetCelebA/dataset4c3s.csv",header=None)
feature = pd.read_csv("DatasetCelebA/list_attr_celeba.csv")

num_righe=3000

#print (dataset.shape)
#print (type(dataset))

#Salvo in datafram le prime 100 righe di dataset
dataframe = dataset.iloc[0:num_righe, 0:48]

#print(feature.shape)
#print(type(feature))

#salvo in feat le prime 100 righe
feat = feature.iloc[0:num_righe,21]
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

#Salvo in X_split le prime 50 righe e 64 colonne per i landmark
X_split = dfconc.iloc[0:num_righe,0:48].values
#Salvo in Y_split le prime 50 righe e la colonna Gender
Y_split = dfconc.iloc[0:num_righe, 48].values #classi

# Splitting del dataset in Training set e Test set rispettivamente 70% e 30%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_split, Y_split, test_size = 0.3, random_state = 0)

print(X_train.shape)
#print(X_train)
print(y_train.shape)
#print(y_train)
print(X_test.shape)
#print(X_test)
print(y_test.shape)
#print(y_test)


#Creazione del classificatore

#classifier = SVC(kernel="linear", C=1)
classifier = SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
   decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
   max_iter=-1, probability=False, random_state=1, shrinking=True, tol=0.001,
   verbose=False)

# classifier = SVC(kernel='sigmoid', gamma=.01, C=1) #al variare di gamma e C
# classifier = SVC(kernel='poly', random_state=1, gamma='auto', C=1) #al variare di gamma e C

# classifier = MLPClassifier(activation='relu', alpha=1, batch_size='auto', beta_1=0.9,
#               beta_2=0.999, early_stopping=False, epsilon=1e-08,
#               hidden_layer_sizes=(100,), learning_rate='constant',
#               learning_rate_init=0.001, max_fun=15000, max_iter=1000,
#               momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
#               power_t=0.5, random_state=None, shuffle=True, solver='adam',
#               tol=0.0001, validation_fraction=0.1, verbose=False,
#               warm_start=False)


#Si allena il classificatore
classifier.fit(X_train, y_train)

# E ora si predice sul Test Set
predicted = classifier.predict(X_test)

#print(predicted)
#print(y_test)

labels = ("Female","Male")
positions = (0,1)

#Stampa dei risultati
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test, cmap="OrRd")
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

print ("\nTempo trascorso: {:.2f}m\n".format((time.time()-start_time)/60))


#Stampa a video
plt.xticks(positions,labels)
plt.yticks(positions,labels)
plt.savefig('ConfusionMatrix.png', bbox_inches='tight')
plt.show()


# Accuratezza
from sklearn.metrics import accuracy_score
print("Accuratezza: ")
print(accuracy_score(y_test, predicted))


