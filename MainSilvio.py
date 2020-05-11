import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import pydotplus
from IPython.display import Image
from IPython.display import display
#caricamento dei due dataset
dataset = pd.read_csv("DatasetCelebA/dataset4c4s.csv",header=None)
feature = pd.read_csv("DatasetCelebA/list_attr_celeba.csv")


dataframe = dataset.iloc[0:100, 0:64]
print(dataframe)


#prendo la colonna delle features riguardante il sesso
feat = feature.iloc[0:100,21]
df_X = pd.DataFrame(feat)


one_hot_data = pd.get_dummies(dataframe.astype(str))

#rinonimo la colonna da Male a Gender
rename = df_X.rename(columns={"Male" : "Gender"}) #-1 donna e 1 maschio

#Concateno i due dataframe per crearne uno
dfconc = pd.concat([one_hot_data, rename], axis=1, sort=False)
print(dfconc)

# The decision tree classifier
clf = tree.DecisionTreeClassifier()
#Training the Decision Tree
clf_train = clf.fit(dfconc, dfconc["Gender"])

# Export/Print a decision tree in DOT format.
print(tree.export_graphviz(clf_train, None))
#Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(dfconc.columns.values),
                                class_names=['Female', 'Male'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
#Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
test2 = graph.write_png("hallo.png");
