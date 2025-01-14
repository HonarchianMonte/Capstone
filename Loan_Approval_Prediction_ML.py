import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing 
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn import metrics



def main():
    data = pd.read_csv("../Capstone/Data/LoanApprovalPrediction.csv")
    
    data.drop(['Loan_ID'],axis=1,inplace=True)

    obj = (data.dtypes == 'object')
    print("Categorical variables:",len(list(obj[obj].index)))

    obj = (data.dtypes == 'object') 
    object_cols = list(obj[obj].index) 
    num_cols = 3  
    num_rows = (len(object_cols) + num_cols - 1) // num_cols 

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
    axes = axes.flatten()
    
    for index, col in enumerate(object_cols):
        y = data[col].value_counts().reset_index()
        y.columns = ['category', 'count']
        max_count = y['count'].max()
        sns.barplot(x='category', y='count', data=y, hue='category', palette="muted", ax=axes[index], legend=False)
        axes[index].tick_params(axis='x', rotation=90)
        axes[index].set_ylabel(col)
        axes[index].set_ylim(0, max_count + 50)
        axes[index].yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=5))
 
    plt.tight_layout(pad=3.0)  
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()
  
    # Label encode the categorical columns
    label_encoders = {}
    for col in object_cols:
        le = preprocessing.LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Print the first 5 rows of the dataset after encoding
    print(data.head(5))

    obj = (data.dtypes == 'object') 
    print("Categorical variables:",len(list(obj[obj].index)))


    plt.figure(figsize=(12,6))
    sns.heatmap(data.corr(),cmap='BrBG',fmt='.2f',linewidths=2,annot=True)
    plt.show()
    
    sns.catplot(x="Gender", y="Married", 
            hue="Loan_Status",  
            kind="bar",  
            data=data)
    plt.show()
    
    for col in data.columns: 
        data[col] = data[col].fillna(data[col].mean())
     # Check for missing values and print the result
    missing_values = data.isna().sum()
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"Column '{col}' has {missing} missing values.")
        else:
            print(f"Column '{col}' has 0.")
    
    # Splitting Data
    X = data.drop(['Loan_Status'], axis=1)
    Y = data['Loan_Status']
    print(X.shape,Y.shape)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,  test_size=0.4, random_state=1)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    
    
    knn = KNeighborsClassifier(n_neighbors=3)
    rfc = RandomForestClassifier(n_estimators = 7, criterion = 'entropy', random_state = 7)

    svc = SVC()
    lc = LogisticRegression(max_iter=5000, solver='liblinear')

        # Cross-validation
    classifiers = [rfc, knn, svc, lc]
    for clf in classifiers:
        scores = cross_val_score(clf, X_train_scaled, Y_train, cv=5)
        print(f"Cross-validation scores for {clf.__class__.__name__}: {scores}")
        print(f"Mean cross-validation score for {clf.__class__.__name__}: {scores.mean():.2f}")

    # Making predictions on the training set
    for clf in classifiers:
        clf.fit(X_train_scaled, Y_train)
        Y_pred = clf.predict(X_train_scaled)
        print("Accuracyscore of ", clf.__class__.__name__,"=", 100*metrics.accuracy_score(Y_train,Y_pred))

    # Making predictions on the testing set 
    for clf in classifiers: 
        clf.fit(X_train, Y_train) 
        Y_pred = clf.predict(X_test_scaled) 
        print("Accuracy score of ", 
            clf.__class__.__name__,"=", 
            100*metrics.accuracy_score(Y_test, Y_pred))
        
        # Prompt user for input
    user_input = {}
    for col in X.columns:
        if col in object_cols:
            unique_values = label_encoders[col].inverse_transform(data[col].unique())
            user_input[col] = [input(f"Enter value for {col} (options: {unique_values}): ")]
        else:
            user_input[col] = [float(input(f"Enter value for {col}: "))]


    user_input_df = pd.DataFrame(user_input)
    
    # Encode categorical variables in user input
    for col in user_input_df.columns:
        if col in object_cols:
            user_input_df[col] = label_encoders[col].transform(user_input_df[col])

    # Make predictions based on user input
    for clf in (rfc, knn, svc, lc):
        prediction = clf.predict(user_input_df)
        prediction_label = "Approved" if prediction[0] == 1 else "Not Approved"
       
    # Get the probability of the prediction
    if hasattr(clf, "predict_proba"):
        probability = clf.predict_proba(user_input_df)[0][prediction[0]]
        print(f"Prediction by {clf.__class__.__name__}: {prediction_label} with probability {probability:.2f}")
    else:
        print(f"Prediction by {clf.__class__.__name__}: {prediction_label}")

if __name__ == "__main__":
    main()