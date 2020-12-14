import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

Train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
TestA_data = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

numerical_cols = Train_data.select_dtypes(exclude = 'object').columns
categorical_cols = Train_data.select_dtypes(include = 'object').columns

## 选择特征列
feature_cols = [col for col in numerical_cols if col not in ['SaleID','name','regDate','creatDate','price','model','brand','regionCode','seller']]
feature_cols = [col for col in feature_cols if 'Type' not in col]

## 提前特征列，标签列构造训练样本和测试样本
X_data = Train_data[feature_cols]
Y_data = Train_data['price']

X_test  = TestA_data[feature_cols]
# plt.hist(Y_data)
# plt.show()
# plt.close()

X_data = X_data.fillna(-1)
X_test = X_test.fillna(-1)

xgbr=xgb.XGBRegressor()#调用XGBRegressor函数‍

xgbr.fit(X_data,Y_data) #拟合

xgbr_y_predict=xgbr.predict(X_test)#预测
#print(xgbr_y_predict.shape)
#print(X_test)

myresult=TestA_data.loc[:,('SaleID')]
myresult=pd.DataFrame(myresult)
myresult.insert(1, 'price', xgbr_y_predict)
print(myresult)
myresult.to_csv("predictions.csv",index=False)
help(xgb.XGBRegressor)