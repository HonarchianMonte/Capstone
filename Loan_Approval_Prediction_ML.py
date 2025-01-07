import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing 
from matplotlib.ticker import MaxNLocator



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
        y.columns = [col, 'count']
        max_count = y['count'].max()
        sns.barplot(x=col, y='count', data=y, palette="muted", ax=axes[index])
        axes[index].set_xticklabels([''] * len(y), rotation=90)
        axes[index].set_ylabel(col)
        axes[index].set_ylim(0, max_count + 50)
        axes[index].yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=5))

    plt.tight_layout(pad=3.0)  
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()
  
    # Label encode the categorical columns
    label_encoder = preprocessing.LabelEncoder()
    obj = (data.dtypes == 'object')
    for col in object_cols:
        data[col] = label_encoder.fit_transform(data[col])

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
            print(f"Column '{col}' has no missing values.")
    
if __name__ == "__main__":
    main()