                                ### THE SPARKS FOUNDATION ###       
### DATA SCIENCE AND BUSINESS ANALYTICS INTERN ###   ### GRIPDECEMBER22 ###   ### DECEMBER 2022 ###
                ### MRIDUL KAPOOR ###       ### mridul.kapoor2002@gmail.com ###
            ### TASK-1 ###      ### PREDICTION USING SUPERVISED MACHINE LEARNING ###

# IMPORTING MODULES
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
data_url = "http://bit.ly/w-data"                                                                       # url of data as in task list
data_file = pd.read_csv(data_url)                                                                       # reading file
print("\nData imported successfully")                                                                   # to check whether data is imported or not

data_file.head()                                                                                        # to print data collected
data_file.isnull().sum()                                                                                # to check for null values or errors in dataset


# DATA PLOTTING
data_file.plot(x = "Hours", y = "Scores", style = "o")

plt.title("HOURS vs SCORE")                                                                             # graph chart title
plt.xlabel("HOURS")                                                                                     # x axis
plt.ylabel("SCORE")                                                                                     # y axis
plt.show()                                                                                              # graph generation


# X AND Y AXIS DATA VALUES
X = data_file.iloc[:, :-1].values  
Y = data_file.iloc[:, 1].values 


# SPLITTING DATA INTO TEST SET AND TRAINING SET 
from sklearn.model_selection import train_test_split                                                    # scikit-learn built-in function
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)


# IMPORTING LINEAR REGRESSION MODEL
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, Y_train)

print("\nTraining complete.")                                                                           # to indicate model is prepared


# CREATING INSTANCE AND CREATING REGRESSION LINE
print("\nRegression Points")
line =regressor.coef_*X+regressor.intercept_                                                            # regression line passing through dots
print(line)                                                                                             # straight line

plt.title("HOURS vs SCORE")                                                                             # graph chart title
plt.xlabel("HOURS")                                                                                     # x axis
plt.ylabel("SCORE")                                                                                     # y axis
plt.scatter(X, Y)                                                                                       # dots genration
plt.plot(X, line);                                                                                      # dotplot graph generation
plt.show()                                                                                              # displays dotplot graph with regression line

print("\nTest Points")
print(X_test)                                                                                           # testing data in hours
Y_pred = regressor.predict(X_test)                                                                      # score prediction


# for comparison purpose only
# COMPARING ACTUAL DATA WITH PREDICTED DATA
data_file = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
data_file                                                                                               # displays comparative table

print("\nTraining score:", regressor.score(X_train, Y_train))                                           # estimating the Training Data and Test Data Score
print("Testing score:", regressor.score(X_test, Y_test))

data_file.plot(kind='line')                                                                             # ploting the line graph to depict the diffrence between the actual and predicted value
plt.title("HOURS vs SCORE")                                                                             # graph chart title
plt.xlabel("HOURS")                                                                                     # x axis
plt.ylabel("SCORE")                                                                                     # y axis
plt.show()


# PREDICTING FOR USER SPECIFIED HOUR INPUT 
# HERE PREDICTING FOR 9.25 HRS/DAY
hours_input = float(input("\nEnter hours student studied: "))
hours = np.array(hours_input).reshape(1, -1)
predict_score = regressor.predict(hours)
print("\nIf the student reads for %0.3f hours then he will score %0.3f"%(hours_input, predict_score[0]))


# FOR MODEL EVALUATION
# PROVIDES ACCURACY OF MODEL
from sklearn import metrics  
print("\nMean Absolute Error:", metrics.mean_absolute_error(Y_test, Y_pred)) 
print("Mean Squared Error:", metrics.mean_squared_error(Y_test, Y_pred))
print("Root mean squared Error:", np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)),"\n")

print("### END ###")