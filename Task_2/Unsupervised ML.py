                                ### THE SPARKS FOUNDATION ###       
    ### DATA SCIENCE AND BUSINESS ANALYTICS INTERN ###   ### GRIPDECEMBER22 ###   ### DECEMBER 2022 ###
                ### MRIDUL KAPOOR ###       ### mridul.kapoor2002@gmail.com ###
            ### TASK-2 ###      ### PREDICTION USING UNSUPERVISED MACHINE LEARNING ###
            
# IMPORTING MODULES
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns

data_file = pd.read_csv("C:/Users/Mridul_Work/Desktop/TSF_GripDecember22_Mridul Kapoor/Task_2/Iris.csv")        # reading file
print("\nData imported successfully")                                                                           # to check whether data is imported or not

print("\nFirst five rows\n",data_file.head())                                                                   # print first five rows of collected data

print("\nLast five rows\n",data_file.tail())                                                                    # print last five rows of collected data


data_file.isnull().sum()                                                                                        # to check for null values or errors in dataset


print(data_file.Species.nunique())                                                                              # to check for classes in given data
print(data_file.Species.value_counts())

# DATA VISUALIZATION

print("\nDot Plot")

sns.set(style = 'whitegrid')                                                                                    # Using Dot Plot
dataset_iris = sns.load_dataset('iris')
axis = sns.stripplot(x ='species',y = 'sepal_length',data = dataset_iris)

plt.title('Iris Dataset')
plt.grid(False)
plt.show()


print("\nCount Plot")

sns.countplot(x='species', data=dataset_iris, palette="OrRd")                                                   # Using Count Plot

plt.title("Count of different species in Iris dataset")
plt.grid(False)
plt.show()


# FINDING OPTIMUM NUMBER OF CLUSTERS FOR K-MEANS
x = data_file.iloc[:,[0,1,2,3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):                                                                                             # driving code
    
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    
    wcss.append(kmeans.inertia_)                                                                                   # appending the WCSS to the list (kmeans.inertia_ returns the WCSS value for an initialized cluster)
    
    print("k: {} ; wcss: {}".format(i,kmeans.inertia_))                                                            # points generation
    

# Plotting the results onto a line graph, allowing us to observe 'The elbow'

plt.figure(figsize=(15,8))
plt.plot(range(1,11),wcss)
plt.title('THE ELBOW METHOD')
plt.xlabel('NUMBER OF CLUSTERS')
plt.ylabel('WCSS')
plt.grid(False)
plt.show()


#APPLYING K-MEANS -- K-MEANS CLASSFIER 

kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)                     # applying K-Means to the dataset 

y_kmeans = kmeans.fit_predict(x)                                                                                      # Returns a label for each data point based on the number of clusters


# VISUALIZING THE CLUSTERS

print("\nWithout Plotting the Centroids of each species")

plt.figure(figsize=(15,8))                                                                                              # providing size of graph 

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='yellow',label='Iris-setosa')                                     # dot plot of setosa
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label='Iris-versicolour')                                  # dot plot of versicolour
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='Iris-virginica')                                   # dot plot of virginica

plt.grid(False)                                                                                                         # to remove gridlines
plt.show()

##Plotting the centroids of the clusters
print("\n After Plotting the Centroids of all species (highlighted in red)")
plt.figure(figsize=(15,8))

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='yellow',label='Iris-setosa')                                     # dot plot of setosa
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label='Iris-versicolour')                                  # dot plot of versicolour
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='Iris-virginica')                                   # dot plot of virginica

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='red',label='Centroids')                  # plotting dots on graph

plt.title('IRIS FLOWER CLUSTERS')                                                                                       # title of graph 
plt.xlabel('SEPAL LENGTH (in cm)')                                                                                      # x axis
plt.ylabel('PETAL LENGTH (in cm)')                                                                                      # y axis
plt.legend()                                                                                                            # legend key

plt.grid(False)                                                                                                         # to remove gridlines
plt.show()
