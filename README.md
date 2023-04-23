# Ex-06-Feature-Transformation

AIM
To read the given data and perform Feature Transformation process and save the data
 to a file

ALGORITHM
STEP 1 : Read the given Data
STEP 2 : Clean the Data Set using Data Cleaning Process
STEP 3 : Apply Feature Transformation techniques to all the features of the data set
STEP 4 : Save the data to the file

CODE
import pandas as pd
df=pd.read_csv('/content/Data_to_Transform.csv')
df.head()

df.isnull().sum()

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew']=np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew']=1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew']=np.sqrt(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer("yeo-johnson")
df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

OUTPUT
 
 ![image](https://user-images.githubusercontent.com/96919035/233819989-3d238aa9-eefd-4576-94a8-b2f76643c5fb.png)
![image](https://user-images.githubusercontent.com/96919035/233820128-35263f4f-2c0b-4eb5-948c-871254d674a3.png)
![image](https://user-images.githubusercontent.com/96919035/233820164-397e97f9-5987-4ffa-98a7-0f27f3f1b9f2.png)
![image](https://user-images.githubusercontent.com/96919035/233820181-fe3f0035-9520-4b3a-8a47-8c2e9586eb80.png)
![image](https://user-images.githubusercontent.com/96919035/233820221-23f82b93-bc3c-4067-852e-ad7cdacd911e.png)
![image](https://user-images.githubusercontent.com/96919035/233820234-a19b18f2-8f59-4356-b66c-a8192b54812c.png)
![image](https://user-images.githubusercontent.com/96919035/233820254-f04acca4-704e-42bf-a9a3-03a23e9ad0cf.png)
![image](https://user-images.githubusercontent.com/96919035/233820544-78f47cce-f766-4e62-a048-5bc98cce08d0.png)
![image](https://user-images.githubusercontent.com/96919035/233820551-6c003555-6b6e-4722-818b-a41c664abb98.png)




RESULT 
Thus the Feature Transformation for the given datasets had been executed successfully

