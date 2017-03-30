## Big Data Analytics Homework-1
####  M10502282 �B����

##### Q&A :

1. �����ݩʹ��c�N�{���������ġH
	- �Q�� Feature importances with forests of trees ���R�c�N�{���A�Y��X���G�yimportances�z�U���̡A�h�P�_���ݩʹ�c�N�{�����������ġC
1. �����ݩʹ��c�N�{�������L�ġH
	- �Q�� Feature importances with forests of trees ���R�c�N�{���A�Y��X���G�yimportances�z�U�C�̡A�h�P�_���ݩʹ�c�N�{���������L�ġC
1. �Τ����k�i�H���U�A�M�w�W�z�����סH
	- Feature importances with forests of trees
1. �z�LPython���ǮM��H�Τ�k�i�H���U�A�����W�����u�@�H - pandas
	- numpy
	- matplotlib.pyplot
	- RandomForestClassifier
	- ExtraTreesClassifier
1. �ҵ{�������L��ĳ�H
	- �Ѯv�ܻ{�u�A�ثe����Ыܦh���R��ƪ��������ΡA���Ʊ�Ѯv�i�H��C�@�I�t�סA��`�J�P���㤶�����ΡA����Ǫ̦Ө��A�|�����U�A���¦Ѯv�C

------------
##### Reference :
1. scikit-learn.org
	- http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
1. Microsoft-Malware-Challenge
	- https://github.com/ManSoSec/Microsoft-Malware-Challenge
------------
##### Code :
```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

mydata = pd.read_csv('/Microsoft-Malware-Challenge/Dataset/train/LargeTrain.csv')

X = np.array(mydata.ix[:,0:1804])
y = np.array(mydata.ix[:,1804:1805]).ravel()

forest = ExtraTreesClassifier(n_estimators=10,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f, indices[f], importances[indices[f]]))

plt.figure()

plt.title("Feature importances",fontsize=16, fontweight='bold')

plt.bar(range(10), importances[indices[0:10]], color="g",align="center")

plt.xlabel('Rank')
plt.ylabel('Impoartance')

plt.xticks(range(10), range(10))

plt.xlim([-1, 10])# set the xlim to xmin, xmax
plt.show()
```

------------

##### Result :
![](https://github.com/Liu-Yi-Chi/Big_Data_Analytics_HW1/blob/master/img/result.PNG)