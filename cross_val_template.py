import matplotlib.pylab as plt
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.patches as patches
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import cross_val_predict
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

my_data = pd.read_csv('../input/train.csv')  
X = my_data.drop(['ID_code','target'], axis=1)
y = my_data['target']
my_data2 = pd.read_csv('../input/test.csv')
X_test = my_data2.drop(['ID_code'], axis=1)
my_data3 = pd.read_csv('../input/sample_submission.csv')

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=0.000001, solver = 'liblinear')
cv = StratifiedKFold(n_splits=5, shuffle = False)

fig1 = plt.figure(figsize=[12,12])
ax1 = fig1.add_subplot(111,aspect = 'equal')

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
i = 1
#ROC curves and confusion matrices
for train,test in cv.split(X,y):
    prediction = clf.fit(X.iloc[train],y.iloc[train]).predict_proba(X.iloc[test])
    fpr, tpr, t = roc_curve(y[test], prediction[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    print('Fold nr.', i, 'AUC is: ',aucs[i-1])
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i= i+1
    print('Confusion matrix:')
    print(confusion_matrix(y.iloc[test], clf.predict(X.iloc[test])))  
    #print(classification_report(y.iloc[test], clf.predict(X.iloc[test])))   
    
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
print('Mean AUC: ', mean_auc)
print()



plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.text(0.32,0.7,'More accurate area',fontsize = 12)
plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()
