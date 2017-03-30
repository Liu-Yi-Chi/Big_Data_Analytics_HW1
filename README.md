## Big Data Analytics Homework-1
####  M10502282 劉奕其

##### Q&A :

1. 哪些屬性對於惡意程式分類有效？
	- 利用 Feature importances with forests of trees 分析惡意程式，若輸出結果『importances』愈高者，則判斷該屬性對惡意程式分類較有效。
1. 哪些屬性對於惡意程式分類無效？
	- 利用 Feature importances with forests of trees 分析惡意程式，若輸出結果『importances』愈低者，則判斷該屬性對惡意程式分類較無效。
1. 用什麼方法可以幫助你決定上述的結論？
	- Feature importances with forests of trees
1. 透過Python哪些套件以及方法可以幫助你完成上面的工作？ - pandas
	- numpy
	- matplotlib.pyplot
	- RandomForestClassifier
	- ExtraTreesClassifier
1. 課程迄今有無建議？
	- 老師很認真，目前為止介紹很多分析資料的相關應用，但希望老師可以放慢一點速度，更深入與完整介紹應用，對於初學者而言，會更有幫助，謝謝老師。

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