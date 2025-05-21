
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt     # Use .pyplot for plotting
import seaborn as sns               # Use 'sns' as the standard alias
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# loading existing data
def load_data(name):
    df = pd.read_csv(name)
    return df

def overview(df):
    print(df.shape)

    # printing missing columns
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing)

    # finding data types
    print(df.dtypes.value_counts())
    print(df.dtypes)

    # basic statistical overview
    print(df.describe())

    # number of unique categories in each column 
    print(df.select_dtypes(include='object').nunique().sort_values(ascending=False))

    # plotting heatmap of correlation matrix
    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix, cmap="coolwarm")
    plt.show()

def fill_na(name, column, value):
    for i in column:
        name[i] = name[i].fillna(value)

def main():
    name = "train.csv"
    df = load_data(name)

    # handling missing values
    fill_na(df, ["MasVnrArea"], 0)
    fill_na(df, ["Electrical"], df["Electrical"].mode()[0])
    fill_na(df, ["GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "MasVnrType", "FireplaceQu", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"], "None")
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    # dropping unwanted columns
    df.drop(["PoolQC", "MiscFeature", "Alley", "Fence", "Id", "GarageArea", "MoSold", "YrSold", "TotalBsmtSF", "TotRmsAbvGrd", "MiscVal", "PoolArea", "3SsnPorch", "BedroomAbvGr"], axis=1, inplace=True)
    print(df.dtypes[df.dtypes == "object"].index.to_list())
    # overview(df)

if __name__ == "__main__":
    main()
