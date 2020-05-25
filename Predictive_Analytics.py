# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV 
from copy import deepcopy
import random

def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """
    cm = ConfusionMatrix(y_true, y_pred)
    correct_pred_sum = cm.trace()
    length_test = len(y_true)#cm.sum()
    accuracy = correct_pred_sum / length_test
    return accuracy

def Recall(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    """
    cm = ConfusionMatrix(y_true, y_pred)
    return(np.diag(cm) / np.sum(cm, axis = 1))

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    """
    cm = ConfusionMatrix(y_true, y_pred)
    return(np.diag(cm) / np.sum(cm, axis = 0))


def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    wcss=[]
    for i in Clusters:
        dist=0
        dist = (i - np.mean(i))**2
        dist = np.sum(dist, axis=1)
        dist = np.sum(dist, axis=0)
        wcss.append(dist)
    return sum(wcss)

def ConfusionMatrix(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    values =len(set(y_true))
    result = y_true*values + y_pred;
    result = np.histogram(result,bins=values*values)
    result = np.reshape(result[0],(values,values))
    return result

def KNN(X_train,X_test,Y_train):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    :rtype: numpy.ndarray
    """
    train_x= pd.DataFrame(X_train)
    test_x=pd.DataFrame(X_test)
    train_x=(train_x-train_x.mean())/train_x.std()
    test_x=(test_x-test_x.mean())/test_x.std()
    def distance(x,train_x,train_y):
        dist = (train_x.to_numpy() - x)**2
        dist = np.sum(dist, axis=1)
        dist=np.sqrt(dist)
        distances=pd.DataFrame()
        distances["class"]=train_y
        distances["dist"]=dist
        distances=distances.sort_values(by='dist', ascending=True)
        return distances
    def assignment(distances,k):
        data=distances[:k].to_numpy() [[1,2],[2,3]]
        word_counter = {}
        for word in data:
            value ={'count':0,'distance':0}
            if word[0] in word_counter:
                word_counter[word[0]]['count'] =  word_counter[word[0]]['count'] + 1
                word_counter[word[0]]['distance'] = word_counter[word[0]]['distance'] + word[1]
            else:
                value['count']=1
                value['distance'] =word[1]
                word_counter[word[0]] = value
        values = sorted(word_counter.items(), key=lambda x: (x[1]['count'],-x[1]['distance']), reverse=True)
        return(values[0][0])
    k=5
    results = np.array([])
    for index in test_x.index:
        distances=distance(test_x.loc[index].values,train_x,Y_train)
        results = np.append(results,assignment(distances,k))
    return results


def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    def giniImpurity(data):
        x, counts = np.unique(data[:, -1], return_counts=True)
        prob = counts / counts.sum()
        giniImpurity = 1 - np.sum(np.power(prob, 2))
        return giniImpurity
    
    def giniWeighted(rightData, leftData):
        giniRight = len(rightData) / (len(rightData) + len(leftData))
        giniLeft = len(leftData) / (len(rightData) + len(leftData))
        giniweighted =  (giniRight * giniImpurity(rightData) 
                          + giniLeft * giniImpurity(leftData))   
        return giniweighted
    
    def getSplits(data, subspace):
        indexes = list(range(data.shape[1] - 1))
        splits = {}
        if subspace and subspace <= len(indexes):
            indexes = random.sample(population=indexes, k=subspace)
        for i in range(len(indexes)):          
            splits[indexes[i]] = np.unique(data[:, indexes[i]])
        return splits

    def findSplit(data, splits):
        giniWeight = 1000
        for index in splits:
            for i in range(len(splits[index])):
                rightData = data[data[:, index] <= splits[index][i]]
                leftData = data[data[:, index] > splits[index][i]] 
                currentGiniWeight = giniWeighted(rightData, leftData)
                if giniWeight - currentGiniWeight >=0:
                    giniWeight = currentGiniWeight
                    splitColumn = index
                    splitValue = splits[index][i]

        return splitColumn, splitValue


    def decisionTree(df, root=0, samples=2, depth=5, random_subspace = None):

        if root == 0:
            global headers
            headers = df.columns
            data = df.values
        else:
            data = df

        if (len(set(data[:, -1])) == 1) or (len(data) < samples) or (root == depth):
            classes, nclasses = np.unique(data[:, -1], return_counts=True)
            return classes[np.argmax(nclasses)]

        root = root + 1

        splitColumn, splitValue = findSplit(data, getSplits(data, random_subspace))
        rightData = data[data[:, splitColumn] <= splitValue]
        leftData = data[data[:, splitColumn] > splitValue]


        if len(rightData) == 0 or len(leftData) == 0:
            classes, nclasses = np.unique(data[:, -1], return_counts=True)
            return classes[np.argmax(nclasses)]

        columnName = headers[splitColumn]
        query = str(columnName) + "," + str(splitValue)
        yes = decisionTree(rightData, root, samples,depth, random_subspace)
        no = decisionTree(leftData, root, samples, depth, random_subspace)
        subTree = {query: []}
        if yes == no:
            subTree = yes
        else:
            subTree[query].append(yes)
            subTree[query].append(no)
        return subTree

    def predict(sample, tree):
        query = list(tree.keys())[0]
        columnName, value = query.split(",")  
        if sample[columnName] >= float(value):
            result = tree[query][1]
        else:
            result = tree[query][0]    
        if isinstance(result, dict):
            return predict(sample, result)    
        else:
            return result



    def decisionTreePred(X_test, tree):
        predictions = X_test.apply(predict, args=(tree,), axis=1)
        return predictions
    
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_train['labels'] = Y_train
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    def randomForest(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
        forest = []
        for i in range(n_trees):
            df_bootstrapped = train_df.sample(n= n_bootstrap,replace=True)
            tree = decisionTree(df_bootstrapped, depth=dt_max_depth, random_subspace=n_features)
            forest.append(tree)
        return forest

    forest = randomForest(X_train, n_trees=10, n_bootstrap=50, n_features=7, dt_max_depth=4)
    pred = {}
    for i in range(len(forest)):
        name = "tree" + str(i)
        pred[name] = decisionTreePred(X_test, tree=forest[i])
    predictions = pd.DataFrame(pred).mode(axis=1)[0].to_numpy()

    return predictions
    
    forest = randomForest(train_df, n_trees=50, n_bootstrap=1000, n_features=7, dt_max_depth=5)
    predictions = randomForestPred(test_df, forest)
    return predictions
    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    X_train -= np.mean(X_train, axis = 0)  
    covarianceM = np.cov(X_train, rowvar = False)
    eigenValues , eigenVectors = np.linalg.eig(covarianceM)
    index = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[index]
    eigenVectors = eigenVectors[:,index]

    totalSum =sum(eigenValues)
    variance = [(i/totalSum)*100 for i in sorted(eigenValues, reverse = True)]
    cumValues = np.cumsum(variance)

#     varianceIndex = next(x for x, val in enumerate(cumValues)if val > 95)
    reducedData = np.dot(X_train, eigenVectors[:,:N])
    return (reducedData)

def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    np.random.seed(42)
  
    centers =X_train[np.random.choice(X_train.shape[0], N, replace=False), :]
    updatedCenters= deepcopy(centers)

    initialCenters = np.zeros(centers.shape) 
    clusters = np.zeros(X_train.shape[0])
    distances = np.zeros((X_train.shape[0],N))

    dist_centers = np.linalg.norm(updatedCenters - initialCenters)

    while dist_centers != 0:
        i = 0;
        while i<N:
            distances[:,i] = np.linalg.norm(X_train - centers[i], axis=1)
            i = i+1;
        initialCenters = deepcopy(updatedCenters)
        clusters = np.argmin(distances, axis = 1)
        j =0;
        while j<N:
            updatedCenters[j] = sum(X_train[clusters == j])/len(X_train[clusters == j])
            j = j+1;
        dist_centers = np.linalg.norm(updatedCenters - initialCenters) 
    df1= pd.DataFrame(X_train)
    df1['clusters']=clusters
    Clusters=[]
    for i in range(N):
        idx = df1['clusters'] == i
        xi=df1[idx] 
        xi=xi.drop(columns = xi.columns[len(xi.columns)-1]).to_numpy()
        Clusters.append(xi)
    return Clusters


def SklearnSupervisedLearning(X_train,Y_train,X_test,Y_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray]
    """
    accuracy =[];
    # Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # SVM model 
    classifierSVM = SVC(kernel='linear',random_state=0)
    classifierSVM.fit(X_train,y_train)
    y_predSVM = classifierSVM.predict(X_test)
    
    #accuracySVM
    accuracySVM = Accuracy(Y_test,y_predSVM)
    accuracy.append(accuracySVM)
    
    # Logistic Regression model  
    classifierLG = LogisticRegression(random_state=0)
    classifierLG.fit(X_train,y_train)
    y_predLG = classifierLG.predict(X_test)
    
    #accuracyLogisticregression
    accuracyLG = Accuracy(Y_test,y_predLG)
    accuracy.append(accuracyLG)
    
    # Decision Treemodel
    classifierDT = DecisionTreeClassifier(criterion='entropy',random_state=0)
    classifierDT.fit(X_train,y_train)
    y_predDT = classifierDT.predict(X_test)
    
    #accuracyDecision
    accuracyDT = Accuracy(Y_test,y_predDT)
    accuracy.append(accuracyDT)
     
    # KNN model
    classifierKNN = KNeighborsClassifier(n_neighbors=5,metric='euclidean')
    classifierKNN.fit(X_train,y_train)
    y_predKNN = classifierKNN.predict(X_test)
   
    #accuracyDecision
    accuracyKNN = Accuracy(Y_test,y_predKNN)
    accuracy.append(accuracyKNN)
    
    fig = plt.figure(figsize=(4*4, 4))

    cmSVM = ConfusionMatrix(Y_test,y_predSVM)
    ax = fig.add_subplot(141)
    cax = ax.matshow(cmSVM)
    plt.title('SVM Classifier',y=1.25)
    fig.colorbar(cax,fraction=0.06, pad=0.1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    cmLG = ConfusionMatrix(Y_test,y_predLG)
    ax1 = fig.add_subplot(142)
    cax1 = ax1.matshow(cmLG)
    plt.title('Logistic Regression',y=1.25)
    fig.colorbar(cax1,fraction=0.06, pad=0.1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    cmDT = ConfusionMatrix(Y_test,y_predDT)
    ax2 = fig.add_subplot(143)
    cax2 = ax2.matshow(cmDT)
    plt.title('Decison Tree',y=1.25)
    fig.colorbar(cax2,fraction=0.06, pad=0.1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    cmKNN = ConfusionMatrix(Y_test,y_predKNN)
    ax3 = fig.add_subplot(144)
    cax3 = ax3.matshow(cmKNN)
    plt.title('KNN Classifier',y=1.25)
    fig.colorbar(cax3,fraction=0.06, pad=0.1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fig.tight_layout()
    plt.show()
    
    return accuracy;

def SklearnVotingClassifier(X_train,Y_train,X_test,Y_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray]
    """
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
   
    classifierEnsemble = VotingClassifier(estimators=[
             ('SVM', SVC(kernel='linear',random_state=0)),
             ('Logistic Regression', LogisticRegression(random_state=0)), 
             ('Decision Tree', DecisionTreeClassifier(criterion='entropy',random_state=0)),
             ('KNN',KNeighborsClassifier(n_neighbors=5,metric='euclidean'))], voting='hard')
    classifierEnsemble = classifierEnsemble.fit(X_train,Y_train)
    y_predEnsemble = classifierEnsemble.predict(X_test)
    
    accuracyEnsemble = Accuracy(Y_test,y_predEnsemble)
    cmEnsemble = ConfusionMatrix(Y_test,y_predEnsemble)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cax1 = ax1.matshow(cmEnsemble)
    plt.title('Ensemble Classifier',y=1.25)
    fig.colorbar(cax1,fraction=0.06, pad=0.1)
    plt.xlabel('Predicted')
    plt.ylabel('True') 
    plt.show()
    
    return [accuracyEnsemble];


def gridSearch(X_train,Y_train):
    parameters = [{'C': [0.1,1,10,20],'kernel': ['linear']}]
  
    grid_searchSVM = GridSearchCV(estimator= SVC(random_state=0), 
                           param_grid = parameters, 
                           scoring ='accuracy')
    grid_searchSVM = grid_searchSVM.fit(X_train,Y_train)
    parametersSVM = grid_searchSVM.cv_results_
 
    ax = plt.axes()
    ax.plot(parameters[0]['C'],parametersSVM['mean_test_score']);
    plt.title('SVM Grid Search')
    plt.xlabel('C Value')
    plt.ylabel('Accuracy')
    plt.show();
    
    hyperparameters = {
        'n_neighbors': [3,5,8,10,15,20],
        'metric':['euclidean']
        }
    grid_searchKNN = GridSearchCV(KNeighborsClassifier(), hyperparameters)
    grid_searchKNN = grid_searchKNN.fit(X_train,Y_train)
    
    parametersKNN = grid_searchKNN.cv_results_
    
    ax1 = plt.axes()
    ax1.plot(hyperparameters['n_neighbors'],parametersKNN['mean_test_score']);
    plt.title('KNN Grid Search')
    plt.xlabel('Neighbors')
    plt.ylabel('Accuracy')
    plt.show();
    
    parameter_grid = {'criterion': ['gini','entropy'] }
    grid_searchDT = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid = parameter_grid)
    grid_searchDT.fit(X_train,Y_train)
    parametersDT = grid_searchDT.cv_results_

    ax2 = plt.axes()
    ax2.plot(parameter_grid['criterion'],parametersDT['mean_test_score']);
    plt.title('Decision Tree Grid Search')
    plt.xlabel('Criterion')
    plt.ylabel('Accuracy')
    plt.show();
    
    return;



    
