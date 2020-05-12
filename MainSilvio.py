import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import pydotplus
from IPython.display import Image
from IPython.display import display
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


#Caricamento dei due dataset
dataframe = pd.read_csv("DatasetCelebA/dataset4c4s.csv",header=None)
feature = pd.read_csv("DatasetCelebA/list_attr_celeba.csv")


print(dataframe)


#Prendo la colonna delle features riguardante il sesso
feat = feature.iloc[0:100,21]
df_X = pd.DataFrame(feat)

#Assegno dei nomi a ciascuna colonna del dataframe assegnandogli inoltre solo valori pari a 0 o 1 (utile per il decision tree, il numero delle colonne aumenta)
#one_hot_data = pd.get_dummies(dataframe.astype(str))

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

#The decision tree classifier
clf = tree.DecisionTreeClassifier()

#Alleno il decision tree
clf_train = clf.fit(X_train, y_train)

#Predico la risposta per il dataset
y_pred = clf.predict(X_test)

#Model Accuracy, valuto il modello
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Stampo un decision tree in formato DOT.
#print(tree.export_graphviz(clf_train, None))

#Creo un decision tree in formato DOT utilizzando GraphViz
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=X_train.columns.values,
                                class_names=['Female', 'Male'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
#Creo il decision tree in formato Graph partendo dal formato DOT
graph = pydotplus.graph_from_dot_data(dot_data)
#Salvo in png il decision tree creato
test2 = graph.write_png("hallo.png");
