import pandas as pd # to import csv and for data manipulation
import matplotlib.pyplot as plt # to plot graph
import seaborn as sns # for intractve graphs
import numpy as np # for linear algebra
import datetime # to dela with date and time
from sklearn import tree

from sklearn.preprocessing import StandardScaler # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split # to split the data
from sklearn.cross_validation import KFold # For cross vbalidation
from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
import warnings
warnings.filterwarnings('ignore')


# Lets Use SMOTE for Sampling
# As I mentioned it is also a type of oversampling but in this the data is not replicated but they are created
#lets start with importing libraries

import pandas as pd
from imblearn.over_sampling import SMOTE
data = pd.read_csv('../input/creditcard.csv')


def data_prepration(x): # preparing data for training and testing as we are going to use different data
    #again and again so make a function
    x_features= x.ix[:,x.columns != "Class"]
    x_labels=x.ix[:,x.columns=="Class"]
    x_features_train,x_features_test,x_labels_train,x_labels_test = train_test_split(x_features,x_labels,test_size=0.3)
    print("length of training data")
    print(len(x_features_train))
    print("length of test data")
    print(len(x_features_test))
    return(x_features_train,x_features_test,x_labels_train,x_labels_test)


## first make a model function for modeling with confusion matrix
def model(model,features_train,features_test,labels_train,labels_test):
    clf= model
    clf.fit(features_train,labels_train.values.ravel())
    pred=clf.predict(features_test)
    cnf_matrix=confusion_matrix(labels_test,pred)
    print("the recall for this model is :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))
    fig= plt.figure(figsize=(6,3))# to plot the graph
    print("TP",cnf_matrix[0,0]) # no of fraud transaction which are predicted fraud 这哥么写错了！！！
    print("FP", cnf_matrix[0, 1])  # no of normal transaction which are predicted fraud
    print("FN",cnf_matrix[1,0]) # no of fraud Transaction which are predicted normal
    print("TN", cnf_matrix[1, 1])  # no. of normal transaction which are predited normal
    sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(labels_test,pred))

os = SMOTE(random_state=0) #   We are using SMOTE as the function for oversampling
# now we can devided our data into training and test data
# Call our method data prepration on our dataset
data_train_X,data_test_X,data_train_y,data_test_y=data_prepration(data)
columns = data_train_X.columns


# now use SMOTE to oversample our train data which have features data_train_X and labels in data_train_y
os_data_X,os_data_y=os.fit_sample(data_train_X,data_train_y)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=["Class"])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of normal transcation in oversampled data",len(os_data_y[os_data_y["Class"]==0]))
print("No.of fraud transcation",len(os_data_y[os_data_y["Class"]==1]))
print("Proportion of Normal data in oversampled data is ",len(os_data_y[os_data_y["Class"]==0])/len(os_data_X))
print("Proportion of fraud data in oversampled data is ",len(os_data_y[os_data_y["Class"]==1])/len(os_data_X))


# Let us first do our amount normalised and other that we are doing above
os_data_X["Normalized Amount"] = StandardScaler().fit_transform(os_data_X['Amount'].reshape(-1, 1))
os_data_X.drop(["Time","Amount"],axis=1,inplace=True)
data_test_X["Normalized Amount"] = StandardScaler().fit_transform(data_test_X['Amount'].reshape(-1, 1))
data_test_X.drop(["Time","Amount"],axis=1,inplace=True)


# Now start modeling
clf= RandomForestClassifier(n_estimators=10)
# train data using oversampled data and predict for the test data
print('随机森林模型的数据')
model(clf,os_data_X,data_test_X,os_data_y,data_test_y)



log_cfl = LogisticRegression()
log_cfl.fit(os_data_X, os_data_y)

rf_cfl = RandomForestClassifier(n_jobs = -1,random_state = 42)
rf_cfl.fit(os_data_X, os_data_y)

dec_cfl = tree.DecisionTreeClassifier()
dec_cfl.fit(os_data_X, os_data_y)


from sklearn.ensemble import VotingClassifier
#投票模型
#voting_cfl = VotingClassifier(
    #    estimators = [('rf_cfl', rf_cfl), ('lt', log_cfl), ('dec_cfl', dec_cfl)],
                #     voting='soft', weights = [1, 1, 1.33])

voting_cfl = VotingClassifier(
        estimators = [('log_cfl', log_cfl), ('rf_cfl', rf_cfl), ('dec_cfl', dec_cfl)],
                     voting='soft')
print('投票模型的数据')
model(voting_cfl,os_data_X,data_test_X,os_data_y,data_test_y)

voting_cfl.fit(os_data_X,os_data_y)

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve

y_pred = voting_cfl.predict(data_test_X)
y_score = voting_cfl.predict_proba(data_test_X)[:,1]

cm = confusion_matrix(os_data_y, y_pred)
print(cm)

from sklearn.model_selection import cross_val_score
for clf, label in zip([ log_cfl, rf_cfl, dec_cfl, voting_cfl], ['logRF', 'Random Forest', 'desicion Tree', 'Ensemble']):
    scores = cross_val_score(clf, os_data_X, os_data_y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))




