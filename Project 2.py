import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Loading the Dataset
dataset=pd.read_csv("E:/project 2/WA_Fn-UseC_-Telco-Customer-Churn.csv")
he=dataset.head(3)

#Checking the Null
nl=dataset.isnull().sum()
#0

#Checking the Duplicates
dc=dataset.duplicated().sum()
#0

#Checking the info
inf=dataset.info()

#checking the shape
sh=dataset.shape
#(7043, 21)

#handling the "TotalCharges" columns
dataset["TotalCharges"]=pd.to_numeric(dataset["TotalCharges"],errors='coerce').fillna(0)

#seeing the updated dataset
inf=dataset.info()
#19  TotalCharges      7043 non-null   float64

#Statistic Calculation on("SeniorCitizen","tenure","MonthlyCharges","TotalCharges")
#Mean
mn_sc=dataset["SeniorCitizen"].mean()
#0.1621468124378816
mn_t=dataset["tenure"].mean()
#32.37114865824223
mn_mc=dataset["MonthlyCharges"].mean()
#64.76169246059918
mn_tc=dataset["TotalCharges"].mean()
#2279.7343035638223

#Median
md_sc=dataset["SeniorCitizen"].median()
#0.0
md_t=dataset["tenure"].median()
#29.0
md_mc=dataset["MonthlyCharges"].median()
#70.35
md_tc=dataset["TotalCharges"].median()
#1394.55

#Mode
mo_sc=dataset["SeniorCitizen"].mode()[0]
#0
mo_t=dataset["tenure"].mode()[0]
#1
mo_mc=dataset["MonthlyCharges"].mode()[0]
#20.05
mo_tc=dataset["TotalCharges"].mode()[0]
#0.0

#Standard Deviation
std_sc=dataset["SeniorCitizen"].std()
#0.3686116056100131
std_t=dataset["tenure"].std()
#24.55948102309446
std_mc=dataset["MonthlyCharges"].std()
#30.090047097678493
std_tc=dataset["TotalCharges"].std()
#2266.7944696890213

#IQR
q1_sc=dataset["SeniorCitizen"].quantile(0.25)
#0.0
q1_t=dataset["tenure"].quantile(0.25)
#9.0
q1_mc=dataset["MonthlyCharges"].quantile(0.25)
#35.5
q1_tc=dataset["TotalCharges"].quantile(0.25)
#398.55

q3_sc=dataset["SeniorCitizen"].quantile(0.75)
#0.0
q3_t=dataset["tenure"].quantile(0.75)
#55.0
q3_mc=dataset["MonthlyCharges"].quantile(0.75)
#89.85
q3_tc=dataset["TotalCharges"].quantile(0.75)
#3786.6

iqr_sc=q3_sc-q1_sc
#0.0
iqr_t=q3_t-q1_t
#46.0
iqr_mc=q3_mc-q1_mc
#54.349999999999994
iqr_tc=q3_tc-q1_tc
#3388.0499999999997

#Checking the Correlation
#corr=dataset[["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]].corr()
#plt.figure(figsize=(4,3))
#sns.heatmap(data=corr,annot=True,cmap='coolwarm',fmt=".2f")
#plt.title("Numeric Feature")

#--------------------------------------------------Data Encoding-------------------------------------------------------#
#One-Hot-Encoding
from sklearn.preprocessing import OneHotEncoder
data=dataset[["gender","Partner","Dependents",
"PhoneService","MultipleLines","InternetService","OnlineSecurity",
"OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
"StreamingMovies","PaperlessBilling","PaymentMethod"]]

ohe=OneHotEncoder(drop="first")
ar=ohe.fit_transform(data)
df=pd.DataFrame(ar.toarray(),columns=ohe.get_feature_names_out(["gender","Partner","Dependents",
"PhoneService","MultipleLines","InternetService","OnlineSecurity",
"OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
"StreamingMovies","PaperlessBilling","PaymentMethod"]))

#Inserting in Original Columns
dataset=pd.concat([dataset,df],axis=1)

#seeing the Original Dataset
he=dataset.head(3)

#Ordinal Encoding
from sklearn.preprocessing import OrdinalEncoder
data=[["Month-to-month","One year","Two year"]]
oen=OrdinalEncoder(categories=data)
dataset['Contract']=oen.fit_transform(dataset[['Contract']])

#Seeing the dataset
dataset["Contract"].head(5)

#Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(dataset["Churn"])
le.transform(dataset["Churn"])
dataset["Churn"]=le.transform(dataset["Churn"])

#Seeing the Dataset
dataset["Churn"].head(4)

#Droping the Unnecessary Columns
dataset=dataset.drop(["customerID"],axis=1)

#seeing the Dataset
inf=dataset.info()

#----------------------------------------------Model Building (Classification focus)------------------------------------#
#Dependent and Independent Variable
x=dataset.drop(columns=["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService",
"OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
"StreamingMovies","PaperlessBilling","PaymentMethod","Churn"])
y=dataset["Churn"]

#Importing the train and test
from sklearn.model_selection import train_test_split

#Making train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=42)

#Performing The Scalling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train_scaled=ss.fit_transform(x_train)
x_test_scaled=ss.transform(x_test)

#----------------------------------------------Logistic Regrression(Classification)------------------------------------#
#Importing the Model
from sklearn.linear_model import LogisticRegression

#Fitting the Model
lr=LogisticRegression(C=10,penalty='l2',solver='liblinear')
lr.fit(x_train_scaled,y_train)

#seeing the Prediction Score
lr_score=lr.score(x_test_scaled,y_test)*100
#81.83108587650815(after Tunning)

#performing the cross-validation 
from sklearn.model_selection import cross_val_score
cv_lr=cross_val_score(LogisticRegression(),x_train_scaled,y_train,cv=5)
cv_lr.sort()
#[78.79325643 79.76929902 80.37300178 80.9228039  81.01153505]

#Hyperperameter Tunning for Logistic Regression
from sklearn.model_selection import GridSearchCV

#Making the parameters
df={'C':[0.001,0.01,0.1,1,10,100],
    'penalty':['l1','l2'],
    'solver':['liblinear','saga']}

#making the Model and fit
gd_lr=GridSearchCV(LogisticRegression(),param_grid=df)
gd_lr.fit(x_train_scaled,y_train)

#Seeing the Best parameters
pr_lr=gd_lr.best_params_
#{'C': 10, 'penalty': 'l2', 'solver': 'liblinear'}

pr_sc=gd_lr.best_score_
#0.8019177274740308

#Calculating the Matrix
#1.Confusssion Matrix
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
cf=confusion_matrix(y_test,lr.predict(x_test_scaled))

#making the Graph
#sns.heatmap(cf,annot=True)

#2. Precision Score
ps_lr=precision_score(y_test,lr.predict(x_test_scaled))*100
#68.0

#3. Recall Score
rs_lr=recall_score(y_test,lr.predict(x_test_scaled))*100
#59.24932975871313

#4. F1_score
f1_lr=f1_score(y_test,lr.predict(x_test_scaled))*100
#63.323782234957015

#Making the Comparision Graph for Matrix
#metrix=['Precision_score','recall_score','f1_score']
#score=[68.0, 59.24932975871313, 63.323782234957015]
#plt.figure(figsize=(6,4))
#sns.barplot(x=metrix,y=score,palette='Blues_d')
#plt.ylim(0,100)
#plt.title('Comparission Graph for matrix')
#plt.xlabel('metrix')
#plt.ylabel('score')
#plt.show()

#Precession Recall Curve
#from sklearn.metrics import PrecisionRecallDisplay
#plt.figure(figsize=(6,4))
#PrecisionRecallDisplay.from_estimator(lr,x_test_scaled,y_test)
#plt.title('precissionrecall score')
#plt.show()
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------Decession-Tree Classification-------------------------------------#
from sklearn.tree import DecisionTreeClassifier

#Making the model and fit
dt=DecisionTreeClassifier(criterion='entropy',max_depth=3, max_features='log2',min_samples_leaf=4,min_samples_split=5,splitter='best')
dt.fit(x_train_scaled,y_train)

#Checking the Acurracy Score
dt_score=dt.score(x_test_scaled,y_test)*100
#77.57274662881476

#performing the Hyperperameter tunnning
from sklearn.model_selection import GridSearchCV

#making the parameters
df={"criterion":['gini','entropy'],'max_depth':[3,4,5,6,8,10],'splitter':['best','random'],
'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4],'max_features':['sqrt','log2','None']}

#making the model and fit
gd_dt=GridSearchCV(DecisionTreeClassifier(),param_grid=df)
gd_dt.fit(x_train_scaled,y_train)

#Seeing the best perameters
dt_pram=gd_dt.best_params_

#perfroming the cross validation
from sklearn.model_selection import cross_val_score
cv_dt=cross_val_score(DecisionTreeClassifier(),x_train_scaled,y_train,cv=5)
cv_dt.sort()
print(cv_dt*100)
#[70.1863354  71.58081705 72.93700089 72.93700089 74.71162378]

#Plotting the tree
#from sklearn.tree import plot_tree
#plt.figure(figsize=(20,20))
#plot_tree(dt)
#plt.show()

#Calculating the metrix
from sklearn.metrics import precision_score,recall_score,f1_score

#1. Precision Score
ps_dt=precision_score(y_test,dt.predict(x_test_scaled))*100
#64.02116402116403

#2. Recall Score
rs_dt=recall_score(y_test,dt.predict(x_test_scaled))*100
#32.43967828418231

#3. F1_score
f1_dt=f1_score(y_test,dt.predict(x_test_scaled))*100
#43.06049822064057

#Making the Comparision Graph for Matrix
#metrix=['Precision_score','recall_score','f1_score']
#score=[64.02116402116403, 32.43967828418231, 43.06049822064057]
#plt.figure(figsize=(6,4))
#sns.barplot(x=metrix,y=score,palette='Blues_d')
#plt.ylim(0,100)
#plt.title('Comparission Graph for matrix')
#plt.xlabel('metrix')
#plt.ylabel('score')
#plt.show()

#Precession Recall Curve
#from sklearn.metrics import PrecisionRecallDisplay
#plt.figure(figsize=(6,4))
#PrecisionRecallDisplay.from_estimator(dt,x_test_scaled,y_test)
#plt.title('precissionrecall score')
#plt.show()
#----------------------------------------------------------------------------------------------------------------------#

#--------------------------------------------------Support-Vector-Machine----------------------------------------------#
from sklearn.svm import SVC

#Making the model and fit
sv=SVC(C=10,gamma='scale',kernel='linear')
sv.fit(x_train_scaled,y_train)

#Checking the score
sv_score=sv.score(x_test_scaled,y_test)*100
#81.76011355571327

#Performing Hyperperameter Tunning
from sklearn.model_selection import GridSearchCV

#making the Perameters
df={'C':[0.1,1,10],'kernel':['linear','rbf'],'gamma':['scale','auto']}

#making the Model and fit
gd_sv=GridSearchCV(SVC(),param_grid=df)
gd_sv.fit(x_train_scaled,y_train)

#seeing the best parameters
sv_pram=gd_sv.best_params_
#{'C': 10, 'gamma': 'scale', 'kernel': 'linear'}

#Performing thr cross validation
from sklearn.model_selection import cross_val_score
cv_sv=cross_val_score(SVC(),x_train_scaled,y_train,cv=5)
cv_sv.sort()
#[78.17213842 79.57371226 79.85803017 80.1242236  80.30168589]

#Calculating the metrix
#1. Precision Score
ps_svc=precision_score(y_test,sv.predict(x_test_scaled))*100
#68.01242236024845

#2. Recall Score
rs_svc=recall_score(y_test,sv.predict(x_test_scaled))*100
#58.71313672922251

#3. F1_score
f1_svc=f1_score(y_test,sv.predict(x_test_scaled))*100
#63.021582733812956

#Making the Comparision Graph for Matrix
#metrix=['Precision_score','recall_score','f1_score']
#score=[68.01242236024845, 58.71313672922251, 63.021582733812956]
#plt.figure(figsize=(6,4))
#sns.barplot(x=metrix,y=score,palette='Blues_d')
#plt.ylim(0,100)
#plt.title('Comparission Graph for matrix')
#plt.xlabel('metrix')
#plt.ylabel('score')
#plt.show()

#Precession Recall Curve
#from sklearn.metrics import PrecisionRecallDisplay
#plt.figure(figsize=(6,4))
#PrecisionRecallDisplay.from_estimator(sv,x_test_scaled,y_test)
#plt.title('precissionrecall score')
#plt.show()
#----------------------------------------------------------------------------------------------------------------------#

#--------------------------------------------------K-Neighbour(Classification)-----------------------------------------#
from sklearn.neighbors import KNeighborsClassifier

#making the Model and fit
knn=KNeighborsClassifier(metric='manhattan',n_neighbors=11,weights='uniform')
knn.fit(x_train_scaled,y_train)

#Checking the Score
kn_score=knn.score(x_test_scaled,y_test)*100
#79.91483321504613

#Performing the Tunning
from sklearn.model_selection import GridSearchCV

#Making the perameters
df={'n_neighbors':[3,5,7,9,11],'weights':['uniform','distance'],'metric':['euclidean','manhattan']}

#making the model
gd_knn=GridSearchCV(KNeighborsClassifier(),param_grid=df)
gd_knn.fit(x_train_scaled,y_train)

#Seeing the best parameters
knn_param=gd_knn.best_params_
#{'metric': 'manhattan', 'n_neighbors': 11, 'weights': 'uniform'}

#Performing Cross validation
from sklearn.model_selection import cross_val_score
cv_knn=cross_val_score(KNeighborsClassifier(),x_train_scaled,y_train,cv=5)
cv_knn.sort()
#[73.64685004 75.42147294 75.48845471 75.77639752 76.75244011]

#Calculating the metrix
#1. Precision Score
ps_knn=precision_score(y_test,knn.predict(x_test_scaled))*100
#63.63636363636363

#2. Recall Score
rs_knn=recall_score(y_test,knn.predict(x_test_scaled))*100
#56.30026809651475

#3. F1_score
f1_knn=f1_score(y_test,knn.predict(x_test_scaled))*100
#59.74395448079659

#Making the Comparision Graph for Matrix
#metrix=['Precision_score','recall_score','f1_score']
#score=[63.63636363636363, 56.30026809651475 , 59.74395448079659]
#plt.figure(figsize=(6,4))
#sns.barplot(x=metrix,y=score,palette='Blues_d')
#plt.ylim(0,100)
#plt.title('Comparission Graph for matrix')
#plt.xlabel('metrix')
#plt.ylabel('score')
#plt.show()

#Precession Recall Curve
#from sklearn.metrics import PrecisionRecallDisplay
#plt.figure(figsize=(6,4))
#PrecisionRecallDisplay.from_estimator(knn,x_test_scaled,y_test)
#plt.title('precissionrecall score')
#plt.show()
#----------------------------------------------------------------------------------------------------------------------#

#------------------------------------------------------Voting-Classifier-----------------------------------------------#
from sklearn.ensemble import VotingClassifier

#Making the list of models
ls=[("lr",LogisticRegression(C=10,penalty='l2',solver='liblinear')),
("svc",SVC(C=10,gamma='scale',kernel='linear',probability=True)),("kn",KNeighborsClassifier(metric='manhattan',n_neighbors=11,weights='uniform'))]

#Making the model
vc=VotingClassifier(ls,voting='soft')
vc.fit(x_train_scaled,y_train)

#Checking the score
vc_score=vc.score(x_test_scaled,y_test)*100
#82.04400283889282

#Cross validation
cv_vc=cross_val_score(vc,x_train_scaled,y_train,cv=5)
#[0.80035492 0.80301686 0.79236912 0.78615794 0.79751332]

#Metrix for voting classifier
ps_vc=precision_score(y_test,vc.predict(x_test_scaled))*100
#70.13422818791946
rs_vc=recall_score(y_test,vc.predict(x_test_scaled))*100
#56.03217158176944
f1_vc=f1_score(y_test,vc.predict(x_test_scaled))*100
#62.295081967213115
#----------------------------------------------------------------------------------------------------------------------#

"""#Comaprision of the Models
Models=['Logistic Regression','Decession-Tree','SVM','-K-Neighbor','Voting-Classifier']
Accuracy=[lr_socre,dt_score,svm_score,kn_score,vc_score]
plt.figure(figsize=(6,4))
sns.barplot(x=Models,y=Accuracy,palette='Blues_d')
plt.ylim(0,100)
plt.title('Comparission Graph For Models')
plt.xlabel('Models')
plt.ylabel('Acurracy')
plt.show()"""