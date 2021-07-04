################### problem1#################################################
import pandas as pd
import numpy as np

#load the dataset
comp_data = pd.read_csv("C:/Users/DELL/Downloads/Decision Tree/Company_Data.csv")

comp_data.isnull().sum()
comp_data.dropna()
comp_data.columns
comp_data = comp_data.drop(["Age"], axis = 1)
comp_data['Sales'].describe()

#data pre-processing
#creat dummies for comp_data
comp_data = pd.get_dummies(comp_data, columns = ["ShelveLoc","Urban","US"])

#converting continous type to categorical
max = comp_data['Sales'].max()
comp_data['Sales'] = pd.cut(comp_data.Sales, bins = [-999 , max/2 , 999] , labels=['low' , 'high'])

comp_data['Sales'].unique()
comp_data['Sales'].value_counts()
colnames = list(comp_data.columns)

predictors = colnames[1:]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(comp_data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

#overfitting issues occurs test accuracy is high and training accuracy is low
#using pruning techniques to overcome this issues

import pandas as pd
import numpy as np

#load the dataset
comp_data = pd.read_csv("C:/Users/DELL/Downloads/Decision Tree/Company_Data.csv")
#Drop the age coloumn
comp_data = comp_data.drop(["Age"], axis = 1)

#data pre-processing
#creat dummies for comp_data
comp_data = pd.get_dummies(comp_data, columns = ["ShelveLoc","Urban","US"])

comp_data['Sales'].unique()
comp_data['Sales'].value_counts()
colnames = list(comp_data.columns)

predictors = colnames[1:]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(comp_data, test_size = 0.3)

# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 5,ccp_alpha= 0.05)
regtree.fit(train[predictors], train[target])

# Prediction
test_pred = regtree.predict(test[predictors])
train_pred = regtree.predict(train[predictors])

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(test[target], test_pred)
r2_score(test[target], test_pred)

# Error on train dataset
mean_squared_error(train[target], train_pred)
r2_score(train[target], train_pred)

# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 3)
regtree2.fit(train[predictors],train[target])

# Prediction
test_pred2 = regtree2.predict(test[predictors])
train_pred2 = regtree2.predict(train[predictors])

# Error on test dataset
mean_squared_error(test[target], test_pred2)
r2_score(test[target], test_pred2)

# Error on train dataset
mean_squared_error(train[target], train_pred2)
r2_score(train[target], train_pred2)

## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3)
regtree3.fit(train[predictors], train[target])

# Prediction
test_pred3 = regtree3.predict(test[predictors])
train_pred3 = regtree3.predict(train[predictors])

# measure of error on test dataset
mean_squared_error(test[target], test_pred3)
r2_score(test[target], test_pred3)

# measure of error on train dataset
mean_squared_error(train[target], train_pred3)
r2_score(train[target], train_pred3)

###################################### random forest #################################
import pandas as pd
import numpy as np

#load the dataset
comp_data = pd.read_csv("C:/Users/DELL/Downloads/Decision Tree/Company_Data.csv")

comp_data.isnull().sum()
comp_data.dropna()
comp_data.columns
comp_data = comp_data.drop(["Age"], axis = 1)
comp_data['Sales'].describe()

#data pre-processing
#creat dummies for comp_data
comp_data = pd.get_dummies(comp_data, columns = ["ShelveLoc","Urban","US"])

#Categories the continous data of Sales column
max = comp_data['Sales'].max()
comp_data['Sales'] = pd.cut(comp_data.Sales, bins = [-999 , max/2 , 999] , labels=['low' , 'high'])

comp_data['Sales'].unique()
comp_data['Sales'].value_counts()
colnames = list(comp_data.columns)

predictors = colnames[1:]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(comp_data, test_size = 0.3)

#building randomforest model
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_clf.fit(train[predictors], train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

#Test accuracy
confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

#Train accuracy
confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))

#now its a right fit model

#################################### problem2 ###############################
#load the dataset
import pandas as pd
import numpy as np

#load the dataset
diabetes_data = pd.read_csv("C:/Users/DELL/Downloads/Decision Tree/Diabetes.csv")

diabetes_data.isnull().sum()
diabetes_data.dropna()
diabetes_data.columns

colnames = list(diabetes_data.columns)

predictors = colnames[:8]
target = colnames[8]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(diabetes_data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

#overfitting issues occurs test accuracy is high and training accuracy is low
#using pruning techniques to overcome this issues

import pandas as pd
import numpy as np

#load the dataset
diabetes_data = pd.read_csv("C:/Users/DELL/Downloads/Decision Tree/Diabetes.csv")

#creating dummy for categorical data
diabetes_data = pd.get_dummies(diabetes_data, columns = [" Class variable"])

colnames = list(diabetes_data.columns)

predictors = colnames[:8]
target = colnames[8]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(diabetes_data, test_size = 0.3)

# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 5,ccp_alpha= 0.05)
regtree.fit(train[predictors], train[target])

# Prediction
test_pred = regtree.predict(test[predictors])
train_pred = regtree.predict(train[predictors])

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(test[target], test_pred)
r2_score(test[target], test_pred)

# Error on train dataset
mean_squared_error(train[target], train_pred)
r2_score(train[target], train_pred)

# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 3)
regtree2.fit(train[predictors],train[target])

# Prediction
test_pred2 = regtree2.predict(test[predictors])
train_pred2 = regtree2.predict(train[predictors])

# Error on test dataset
mean_squared_error(test[target], test_pred2)
r2_score(test[target], test_pred2)

# Error on train dataset
mean_squared_error(train[target], train_pred2)
r2_score(train[target], train_pred2)

## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3)
regtree3.fit(train[predictors], train[target])

# Prediction
test_pred3 = regtree3.predict(test[predictors])
train_pred3 = regtree3.predict(train[predictors])

# measure of error on test dataset
mean_squared_error(test[target], test_pred3)
r2_score(test[target], test_pred3)

# measure of error on train dataset
mean_squared_error(train[target], train_pred3)
r2_score(train[target], train_pred3)

###################################### random forest #################################
import pandas as pd
import numpy as np

#load the dataset
diabetes_data = pd.read_csv("C:/Users/DELL/Downloads/Decision Tree/Diabetes.csv")

#creating dummy for categorical data
diabetes_data = pd.get_dummies(diabetes_data, columns = [" Class variable"])

colnames = list(diabetes_data.columns)

predictors = colnames[:8]
target = colnames[8]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(diabetes_data, test_size = 0.3)

#building randomforest model
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_clf.fit(train[predictors], train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

#Test accuracy
confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

#Train accuracy
confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))

#now its a right fit model

#################################### problem3 #############################################
import pandas as pd
import numpy as np

#load the dataset
fraud_check = pd.read_csv("C:/Users/DELL/Downloads/Decision Tree/Fraud_check.csv")

fraud_check.isnull().sum()
fraud_check.dropna()
fraud_check.columns
fraud_check['Taxable.Income'].describe()

#data pre-processing
#creat dummies for comp_data
fraud_check = pd.get_dummies(fraud_check, columns = ["Undergrad", "Marital.Status", "Urban"])


#converting continous type to categorical
max = fraud_check['Taxable.Income'].max()
fraud_check['Taxable.Income'] = pd.cut(fraud_check['Taxable.Income'], bins = [-999 , 30000 , 99999] , labels=['Risky' , 'Good'])

fraud_check['Taxable.Income'].unique()
fraud_check['Taxable.Income'].value_counts()
colnames = list(fraud_check.columns)

predictors = colnames[1:]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(fraud_check, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

#overfitting issues occurs test accuracy is high and training accuracy is low
#using pruning techniques to overcome this issues

import pandas as pd
import numpy as np

#load the dataset
fraud_check = pd.read_csv("C:/Users/DELL/Downloads/Decision Tree/Fraud_check.csv")

#data pre-processing
#creat dummies for comp_data
fraud_check = pd.get_dummies(fraud_check, columns = ["Undergrad", "Marital.Status", "Urban"])

colnames = list(fraud_check.columns)

predictors = colnames[1:]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(fraud_check, test_size = 0.3)

# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 5,ccp_alpha= 0.05)
regtree.fit(train[predictors], train[target])

# Prediction
test_pred = regtree.predict(test[predictors])
train_pred = regtree.predict(train[predictors])

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(test[target], test_pred)
r2_score(test[target], test_pred)

# Error on train dataset
mean_squared_error(train[target], train_pred)
r2_score(train[target], train_pred)

# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 3)
regtree2.fit(train[predictors],train[target])

# Prediction
test_pred2 = regtree2.predict(test[predictors])
train_pred2 = regtree2.predict(train[predictors])

# Error on test dataset
mean_squared_error(test[target], test_pred2)
r2_score(test[target], test_pred2)

# Error on train dataset
mean_squared_error(train[target], train_pred2)
r2_score(train[target], train_pred2)

## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3)
regtree3.fit(train[predictors], train[target])

# Prediction
test_pred3 = regtree3.predict(test[predictors])
train_pred3 = regtree3.predict(train[predictors])

# measure of error on test dataset
mean_squared_error(test[target], test_pred3)
r2_score(test[target], test_pred3)

# measure of error on train dataset
mean_squared_error(train[target], train_pred3)
r2_score(train[target], train_pred3)
#now its a right fit model
###################################### random forest #################################
import pandas as pd
import numpy as np

#load the dataset
fraud_check = pd.read_csv("C:/Users/DELL/Downloads/Decision Tree/Fraud_check.csv")

fraud_check.isnull().sum()
fraud_check.dropna()
fraud_check.columns

#data pre-processing
#creat dummies for comp_data
fraud_check = pd.get_dummies(fraud_check, columns = ["Undergrad","Marital.Status","Urban"])

#Categories the continous data of Sales column
max = fraud_check['Taxable.Income'].max()
fraud_check['Taxable.Income'] = pd.cut(fraud_check['Taxable.Income'], bins = [-999 ,30000 , 9999999] , labels=['low' , 'high'])

fraud_check['Taxable.Income'].unique()
fraud_check['Taxable.Income'].value_counts()
colnames = list(fraud_check.columns)

predictors = colnames[1:]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(fraud_check, test_size = 0.3)

#building randomforest model
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_clf.fit(train[predictors], train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

#Test accuracy
confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

#Train accuracy
confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))

#now its a right fit model

############################ problem4 ########################################
########################################Problem 4###########################################
#load the data

import pandas as pd
import numpy as np
HR_data = pd.read_csv("C:/Users/DELL/Downloads/Decision Tree/HR_DT.csv")

#dummy values
HR_data = pd.get_dummies(HR_data, columns = ["Position of the employee"])

#random Forest technique

# Splitting data into training and testing data set
colnames = list(HR_data.columns)
target = colnames[1]
predictors = colnames[:1]+colnames[2:]

from sklearn.model_selection import train_test_split
train, test = train_test_split(HR_data, test_size = 0.3)

#building the random forest model
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=50)

rf_clf.fit(train[predictors], train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

#accuracy on test data
confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

#accuracy on train data
confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))

#create a dataframe of customers claim details
#filling Region Manager as 1 anyways we convert it to dummy here by saving computation
customer_claim_list = [[1, 5.0 , 70000]]

customer_claim = pd.DataFrame(customer_claim_list , columns= ["Position of the employee" , "no of Years of Experience of employee", " monthly income of employee"])

#combining and concatenating 2 dataframes
df = [test , customer_claim]
test = pd.concat(df)

#filling all na by 0
test = test.fillna(0)

#predicting using testing data where customer claims present at last row of test
preds = rf_clf.predict(test[predictors])

#storing predicted values in seperate column
test["predicted salary"] = preds

#accessing predicted value
test.iloc[59,13]

# predicted as 67938 , candidate claimed 70000 which is almost close.
#it can be assumed that candidate is genuine

###########################################END#####################################


















