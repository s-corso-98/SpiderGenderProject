import pandas as pd
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split # Import train_test_split function
import matplotlib.pyplot as plt
import time


start_time = time.time()


#Caricamento dei due dataset
dataframe = pd.read_csv("DatasetCelebA/set5c4s.csv", header=None)
feature = pd.read_csv("DatasetCelebA/list_attr_celeba.csv")


print(dataframe)

#Prendo la colonna delle features riguardante il sesso
feat = feature.iloc[0:100,21]
df_X = pd.DataFrame(feat)

#Rinonimo la colonna da Male a Gender
rename = df_X.rename(columns={"Male" : "Gender"}) #-1 donna e 1 maschio

#Concateno i due dataframe per crearne uno
dfconc = pd.concat([dataframe, rename], axis=1, sort=False)
print(dfconc)

#Ottengo feature variables
feature_cols = list(dfconc.columns.values)
X = feature_cols[1:len(feature_cols)-1]
X = dfconc[X]
print("X:",X)

#Ottengo target variables
y = dfconc.Gender
print("y:",y)

#Divido il dataframe in train e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#NearestNeighbors classifier
classifier = KNeighborsClassifier(n_neighbors=5)

#Alleno il classificatore
clf_train = classifier.fit(X_train, y_train)

#Predico la risposta per il dataset
y_pred = classifier.predict(X_test)

#Model Accuracy, valuto il modello
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print ("\nTempo trascorso: {:.2f}m\n".format((time.time()-start_time)/60))


labels = ("Female","Male")
positions = (0,1)

#Stampa dei risultati
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, y_pred)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

#Stampa a video
plt.xticks(positions,labels)
plt.yticks(positions,labels)
plt.savefig('ConfusionMatrix5c4s.png', bbox_inches='tight')
plt.show()


#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

#CALCOLO ERRORE MEDIO PER K
#error = []

#for i in range(1,40):
#    knn= KNeighborsClassifier(n_neighbors=i)
#    knn.fit(X_train, y_train)
#    pred_i = knn.predict(X_test)
#    error.append(np.mean(pred_i != y_test))


#plt.figure(figsize=(12,6))
#plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
#plt.title('Error rate k value')
#plt.xlabel('K value')
#plt.ylabel('Mean error')
#plt.show()

