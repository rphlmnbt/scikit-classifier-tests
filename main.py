from AdaBoostClassifier import AdaBoost
from DecisionTreeClassifier import DecisionTree
from GradientBoostClassifier import GradientBoost
from KNeighborsClassifier import KNeighbors
from LogisticRegression import Logistic
from RandomForestClassifier import RandomForest
from SupportVectorClassifier import SupportVector
from input import *

test1 = AdaBoost().predict(inputArr)
test2 = DecisionTree().predict(inputArr)
test3 = GradientBoost().predict(inputArr)
test4 = KNeighbors().predict(inputArr)
test5 = Logistic().predict(inputArr)
test6 = RandomForest().predict(inputArr)
test7 = SupportVector().predict(inputArr)
