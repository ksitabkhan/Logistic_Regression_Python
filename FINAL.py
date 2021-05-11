import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def accuracy_functn(Y_predicted, y):  
    score = sum(Y_predicted == y) / len(y)
    return score


class Logistic_Multiclass(object):

    def __init__(Logreg, alpha=0.01, n_iteration=100):  
        Logreg.theta=[]
        Logreg.alpha = alpha  
        Logreg.n_iter = n_iteration

    def sigmoid(Logreg, x):  
        value = 1 / (1 + np.exp(-x))
        return value

    def fit(Logreg, X,y):  
        X = np.insert(X, 0, 1, axis=1)
        length_pred = len(y)
        for i in np.unique(y):
            
            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros(X.shape[1])
            for val in range(Logreg.n_iter):
                z = X.dot(theta)
                h = Logreg.sigmoid(z)
                gradient_value = np.dot(X.T, (h - y_onevsall)) / length_pred
                theta -= Logreg.alpha * gradient_value
            Logreg.theta.append((theta, i))

        return Logreg

    def predict(Logreg, X):  
        X = np.insert(X, 0, 1, axis=1)
        X_predicted = [max((Logreg.sigmoid(i.dot(theta)), c) for theta, c in Logreg.theta)[1] for i in X]
        return X_predicted



def Scalar_function(X):
    for j in range(X.shape[1]):
        for i in range(X.shape[0]):
            #print(X[i][j])
            X[i, j] = (X[i, j] - np.mean(X[:][j])) / np.std(X[:][j])
    return X



def LogisticReg_python(X_train, X_test, y_train, y_test):
    # Running Logistic Regression algorithm
    classifier = LogisticRegression(solver='lbfgs', random_state=0, max_iter=10000)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return y_pred

if __name__ == "__main__":
    colnames = ['calorific_value', 'nitrogen', 'turbidity', 'style', 'alcohol', 'sugars', 'bitterness', 'beer_id', 'colour', 'degree_of_fermentation']
    df1 = pd.read_csv(r'beer.txt', sep="\t", header=None,names=colnames)
    new_cols = ['calorific_value', 'nitrogen', 'turbidity', 'alcohol', 'sugars', 'bitterness', 'beer_id', 'colour', 'degree_of_fermentation' ]

    X = df1.loc[:, new_cols]
#    Scalar_function(X)
    y_data = df1.iloc[:, 3]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    sys.stdout = open("Logistic_Algo_Accuracy.txt", "w")
    print("Generating report for accuracy of multiple iterations of Logistic algorithm with different samples.")
    for i in range(1,11):
        print("\n\n ------------------Iteration # : ", i,"------------------")
        X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.33)
        logi = Logistic_Multiclass(n_iteration=30000).fit(X_train, y_train)
        Y_predicted = logi.predict(X_test)
        DF2 = pd.DataFrame(Y_predicted)
        csv_name = "Predicted_MyAlgo"+str(i)
        DF2.to_csv(csv_name)
        score1 = round(accuracy_functn(Y_predicted,y_test)*100,2)
        print("The accuracy of the developed model is ", score1,"%")

        y_pred = LogisticReg_python(X_train, X_test, y_train, y_test)
        print("Accuracy of Logistic Regression using sklearn is : ", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
        DF3 = pd.DataFrame(y_pred)
        csv_name = "Predicted_SKLearnAlgo" + str(i)
        DF3.to_csv(csv_name)
    sys.stdout.close()