import pandas as pd # to import csv and for data manipulation
import matplotlib.pyplot as plt # to plot graph
import seaborn as sns # for intractve graphs
import numpy as np # for linear algebra
import datetime # to dela with date and time
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import pandas as pd
from imblearn.over_sampling import SMOTE

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

def plot_roc(fpr, tpr):#画Roc-curve
    plt.plot(fpr, tpr, label = 'ROC curve', linewidth = 2)
    plt.plot([0,1],[0,1], 'k--', linewidth = 2)
    plt.xlim([0.0,0.001])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

def plot_precision_recall(recall, precision):#画precision和recall关系
    plt.step(recall, precision, color = 'b', alpha = 0.2,
             where = 'post')
    plt.fill_between(recall, precision, step ='post', alpha = 0.2,
                 color = 'b')

    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0,1])
    plt.ylim([0.0,1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.show()
# Lets Use SMOTE for Sampling
# As I mentioned it is also a type of oversampling but in this the data is not replicated but they are created
#lets start with importing libraries





def data_prepration(x,label): # preparing data for training and testing as we are going to use different data
    #again and again so make a function
    x_features= x.ix[:,x.columns != label]
    x_labels=x.ix[:,x.columns==label]
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
   # plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(labels_test,pred))

    #fpr, tpr, threshold = roc_curve(y_test, y_score)
    #y_score = clf.decision_function(features_test)
    y_score = clf.predict_proba(features_test)[:, 1]

    fpr, tpr, t = roc_curve(labels_test, y_score)
  #  plot_roc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(labels_test, y_score)
   # plot_precision_recall(precision, recall)

if __name__ == '__main__':
    data = pd.read_csv('../input/creditcard.csv')
    #画箱体图
    plt.figure(figsize = (12, 6))
    my_pal = {0: 'deepskyblue', 1: 'deeppink'}
    ax = sns.boxplot(x = 'Class', y = 'Amount', data = data, palette = my_pal)
    ax.set_ylim([0, 300])
    plt.title('Boxplot Amount vs Class')
    plt.show()
    #开始特征处理，建模
    os = SMOTE(random_state=0)
    data_train_X,data_test_X,data_train_y,data_test_y=data_prepration(data,'Class')
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
    print('xgboost模型的数据')
    #clf= RandomForestClassifier(n_estimators=10)
    clf = xgb.XGBClassifier(n_jobs = -1, n_estimators = 200)
    # train data using oversampled data and predict for the test data
    model(clf,os_data_X,data_test_X,os_data_y,data_test_y)




    log_cfl = LogisticRegression()
    #log_cfl.fit(os_data_X, os_data_y)
    rf_cfl = RandomForestClassifier(n_jobs = -1,random_state = 42)
    #rf_cfl.fit(os_data_X, os_data_y)
    xgb_cfl = xgb.XGBClassifier(n_jobs = -1, n_estimators = 200)
    #xgb_cfl.fit(os_data_X, os_data_y)

    print("开始gridcv 找逻辑回归最佳参数\n")
    param_grid = {
                'penalty' : ['l1','l2'],
                'class_weight' : ['balanced', None],
                'C' : [0.1, 1, 10, 100]
                }
    CV_log_cfl = GridSearchCV(estimator = log_cfl, param_grid = param_grid , scoring = 'recall', verbose = 1, n_jobs = -1)
    CV_log_cfl.fit(os_data_X, os_data_y.values.ravel())
    best_parameters = CV_log_cfl.best_params_
    print('The best parameters for using this model is', best_parameters)

    print("开始gridcv 找随机森林最佳参数\n")
    param_grid = {
                'n_estimators': [100, 200, 500],
                'max_features': [2, 3],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10]
                }
    CV_rf_cfl = GridSearchCV(estimator = rf_cfl, param_grid = param_grid , scoring = 'recall', verbose = 1, n_jobs = -1)
    CV_rf_cfl.fit(os_data_X, os_data_y.values.ravel())
    best_parameters = CV_rf_cfl.best_params_
    print('The best parameters for using this model is', best_parameters)


    print("开始gridcv 找xgboost最佳参数\n")
    param_grid = {
                'n_estimators': [100, 200, 300, 400]
                  }
    CV_xgb_cfl = GridSearchCV(estimator = xgb_cfl, param_grid = param_grid, scoring ='recall', verbose = -1,n_jobs = -1)
    CV_xgb_cfl.fit(os_data_X, os_data_y.values.ravel())
    best_parameters = CV_xgb_cfl.best_params_
    print("The best parameters for using this model is", best_parameters)

    print("开始投票模型\n")
    from sklearn.ensemble import VotingClassifier

    voting_cfl = VotingClassifier(
            estimators = [('log_cfl', log_cfl), ('rf_cfl', rf_cfl), ('xgb_cfl', xgb_cfl)],
                         voting='soft')
    print('投票模型的数据')
    model(voting_cfl,os_data_X,data_test_X,os_data_y,data_test_y)