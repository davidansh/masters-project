import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from prettytable import PrettyTable
import seaborn as sns

all_files = glob.glob("client dataset/*.csv")

li = []
for file in all_files:
    df = pd.read_csv(file)
    li.append(df.drop('Code', axis=1))
    
final_df = li[0]
for i in li[1:]:
    final_df = pd.merge(final_df, i, on = ['Entity', 'Year'])

countries = final_df["Entity"].unique()
countries = ['Africa', 'Asia Pacific', 'CIS', 'China', 'Europe', 'France',
       'Germany', 'Italy', 'Mexico', 'Portugal', 'United States', 'World']

#a = final_df.loc[final_df['Entity'] == countries[0]]
#matrix = a.corr()

from sklearn.preprocessing import PolynomialFeatures
for i in countries:
    a = final_df.loc[final_df['Entity'] == i]


    transformer = PolynomialFeatures(degree=3, include_bias=False)
    x = a["Year"]
    x = x.to_numpy().reshape(-1,1)
    y = a[["Electricity from hydro (TWh)_x","Electricity from wind (TWh)_x","Electricity from solar (TWh)_x",
           "Electricity from other renewables including bioenergy (TWh)"]]
    labels = y.columns.to_list()
    y = y.to_numpy()
    transformer.fit(x)
    x_ = transformer.transform(x)
    plt.figure(figsize=(15,15))
    
    model = LinearRegression().fit(x_, y)
    ypred = model.predict(x_)
    plt.plot(x,ypred)
    plt.xlabel("Years")
    plt.legend(labels, loc ="best")
    plt.title(f"{i} 3 Degree Regression for Electricity from Renewable Energies")
    plt.savefig(f"{i} Regression3degree.png", bbox_inches='tight')
    plt.show()
    pred = model.predict((PolynomialFeatures(degree=3, include_bias=False).fit_transform(np.array(2021).reshape(-1,1))))
    myTable = PrettyTable(["category","country", "Prediction for 2021"])
    for j,k in enumerate(labels):
        myTable.add_row([f"{k}", f"{i}" , f"{pred[0][j]}"])
    print(myTable)
    
    corr_matrix = np.corrcoef(y.T).round(decimals=2)
    plt.figure(figsize=(10,10))
    sns.heatmap(corr_matrix, annot=True, xticklabels = ['Hydro', 'Wind', 'Solar', 'BioEnergy'], 
                yticklabels = ['Hydro', 'Wind', 'Solar', 'BioEnergy'])
    plt.savefig(f"{i} Correlation.png", bbox_inches='tight')
    plt.show()








