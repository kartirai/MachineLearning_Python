# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:38:51 2020

@author: raiajay
"""


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = pd.DataFrame(iris.data[:,:],columns = iris.feature_names[:])
y = pd.DataFrame(iris.target,columns = ['Species'])
tree = DecisionTreeClassifier(max_depth = 2)
tree.fit(X,y)

from sklearn.tree import export_graphviz
export_graphviz(
            tree,
            out_file =  "myTreeName.dot",
            feature_names = list(X.columns),
            class_names = iris.target_names,
            filled = True,
            rounded = True)
from subprocess import call
call(['dot', '-T', 'png', 'myTreeName.dot', '-o', 'myTreeName.png'])