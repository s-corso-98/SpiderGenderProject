# SpiderGender

An image processing project to recognize people gender, based on the "pose estimation" algorithm.

## Getting Started
Just clone the github repository to get all the files you need to execute the code.


### Prerequisites

All you need to install is Python 3.7 (we use Anaconda3 environment but it's the same) and these libraries:
* Matplotlib
* Pandas
* pydotplus
* Sklearn


## Running and tests

1. calcoloragnatela.py
* Modify to change pose estimation configuration
--- 
cerchi = 4
fetteQ = 4 
---

* Modify range value to change the number of photos to get processed
---
dizionario = [ [ 0 for y in range(s1)] for x in range(202599)]
dizionario_str = ['' for xx in range(202599)]
---

* Modify these paths to execute the algorithm on other datasets
---
immagini = os.listdir('C:/Users/Administrator/Desktop/CelebA/img_align_celeba/img_align_celeba')
im2 = "C:/Users/Administrator/Desktop/CelebA/img_align_celeba/img_align_celeba/"+str(img)
---

*Modify this path to save the dataset properly
---
numpy.savetxt("DatasetCelebA/dataset4c4s.csv", dizionario, fmt='%i', delimiter=",")

For example if you executed the algorithm based on 5c3s configuration the path will be:
"DatasetCelebA/dataset5c3s"
---

2. Main.py
CelebA dataset has an unbalanced number of male and females so when running the code:
Type "1" to execute tests on the balanced dataset (based on 4c4s configuration)
Type "0" to execute tests on the unbalanced datased (based on the configuration of the given path)

*Modify ReadCSV() function to change dataset to tests
---
Change path:
dataframe = pd.read_csv("DatasetCelebA/dataset4c4s.csv", header=None)

The range change according to the number of rows of the read dataset:
feat = feature.iloc[0:202599, 21]
---

*Uncomment the algorithm that you wanna test:
---
In this case we are executing only the DecisionTree:
MyDecisionTree()
#MyNearestNeighbors()
#MySupportVectorMachine()
---


## Built With

* [Pycharm](https://www.jetbrains.com/pycharm/) - Python IDE
* [Anaconda3](https://www.anaconda.com/) - Package Management


## Authors

* **Paolo Cantarella** - *Initial work* - [Cantarella Paolo](https://github.com/Pcantarella7)
* **Silvio Corso** - *Initial work* - [s-corso-98](https://github.com/s-corso-98)
* **Carmine Tramontano** - *Initial work* - [carminet94](https://github.com/carminet94)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
