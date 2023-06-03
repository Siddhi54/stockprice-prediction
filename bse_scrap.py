from bsedata.bse import BSE
import pandas as pd
import csv
import pickle
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#import models for prediction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score



class bse_scrape:
    def __init__(self):
       #BSE is bse self API
       self.b = BSE()
       self.security_comapny=[[]]
    
    def get_security_code(self):
        #download the data from bse in form of csv and modify it drop some columns -Equity.csv - https://www.bseindia.com/corporates/List_Scrips.html
        file_data = []
        with open (r"./Eqity_modify.csv",'r',encoding="ISO-8859-1") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                file_data.append(row)
        tickers = ['ABB','HDFC','BIRLACABLE','CHAMBLFERT','HEROMOTOCO','BPCL','HIRECT','KANPRPLA','DHANUKA' ]
        codes = []
        

        counter = 0
        counter1 = 0
        for ticker in tickers:
            counter = 0
            for row in file_data:
                if row[2] == ticker:
                    codes.append(str(file_data[counter][1]))
                    self.security_comapny.append([])
                    self.security_comapny[counter1].append(file_data[counter][0])
                    self.security_comapny[counter1].append(file_data[counter][1])
                    counter1+=1
                    break
                counter+=1   
        return codes 
    
    def get_securityCode_by_company(self, company):
        security_code =""
        for i in range(0,len(self.security_comapny)):
            if company in self.security_comapny[i]:
                security_index = i
                security_code =  self.security_comapny[security_index][0]
                break
        return security_code
    
    def get_databysecurity(self, security_code):
        #take 12m data of 1 selected company
        bse_data_analysis = self.b.getPeriodTrend(security_code,'12M')
        print(type(bse_data_analysis))
        data = pd.DataFrame(bse_data_analysis)
        
        data.columns = ['Date', 'Close', 'Volume']
        data['DOB'] = pd.to_datetime(data.Date)
        #sorting date format
        data['Date_fin'] = data['DOB'].dt.strftime('%Y-%m-%d')
        data['Date'] = data['Date_fin'].values
        data = data.drop(columns =['Date_fin','DOB'])
        print(data['Date'])

        # Reindex data using a DatetimeIndex
        data.set_index(pd.DatetimeIndex(data['Date']), inplace=True)
        # Keep only the 'Adj Close' Value
        df = data[['Close']]
        # Re-inspect data
        print(df)

        #moving avg of 10 days
        mvg = data.Close.rolling(10).mean()
        data['ema'] = mvg
        print(data['ema'].head(20))

        #we set date to index so drop extra occured date column from table
        data = data.drop(columns =['Date'])

        #make csv
        data.to_csv("Data.csv")
        print(data.info())

        # Drop the first n-rows
        data = data.iloc[10:]
        print(data.head(20))

        return data
    
    def train_test(self,data):
        #traing data testing data
        x_train,x_test, y_train, y_test = train_test_split(data[['Close']], data[['ema']], test_size = 0.2, random_state=0)
        print (x_train)
        print (x_test)
        print (y_train)
        print (y_test)
        y_test.columns=['y_test']
        # Test set
        print(x_test.describe())
        # Train set
        print(x_train.describe())
        
    
        #training the model using linear reg
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)
        print (regressor.coef_)
        print (regressor.intercept_)
        #prediction
        y_pred_lr = regressor.predict(x_test)
        print(y_pred_lr)
        sqrt_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        abs_lr = mean_absolute_error(y_test, y_pred_lr)
        score_lr = r2_score(y_test, y_pred_lr)
        print("linear:",sqrt_lr, abs_lr, score_lr)


        #training the model using lasso reg
        ls = Lasso()
        ls.fit(x_train, y_train)
        print (ls.coef_)
        print (ls.intercept_)
        #prediction
        y_pred_ls = ls.predict(x_test)
        print(y_pred_ls)
        sqrt_ls = np.sqrt(mean_squared_error(y_test, y_pred_ls))
        abs_ls = mean_absolute_error(y_test, y_pred_ls)
        score_ls = r2_score(y_test, y_pred_ls)
        print("lasso:",sqrt_ls, abs_ls, score_ls)


        #training the model using ridge reg
        rr = Ridge()
        rr.fit(x_train, y_train)
        print (rr.coef_)
        print (rr.intercept_)
        #prediction
        y_pred_rr = rr.predict(x_test)
        print(y_pred_rr)
        sqrt_rr = np.sqrt(mean_squared_error(y_test, y_pred_rr))
        abs_rr = mean_absolute_error(y_test, y_pred_rr)
        score_rr = r2_score(y_test, y_pred_rr)
        print("ridge:", sqrt_rr, abs_rr, score_rr)

    
        print("lasso:",sqrt_ls, abs_ls, score_ls)
        print("linear:",sqrt_lr, abs_lr, score_lr)
        print("ridge:", sqrt_rr, abs_rr, score_rr)

        #find mean
        pred1 = y_pred_lr.mean()
        pred2 = y_pred_ls.mean()
        pred3 = y_pred_rr.mean()
        print(pred1,pred2,pred3)
        
        #making model
        pickle.dump(regressor, open('Linearmodel.pkl', 'wb'))
        pickle.dump(ls, open('Lassomodel.pkl', 'wb'))
        pickle.dump(rr, open('Ridgemodel.pkl', 'wb'))
        
        #prediction dataframe of lr
        Linearreg = pd.DataFrame(y_test['y_test'])
        Linearreg['y_pred_lr'] = y_pred_lr
        print(Linearreg['y_test'])
        print(Linearreg['y_pred_lr'])

        #prediction dataframe of ls
        Lassoreg = pd.DataFrame(y_test['y_test'])
        Lassoreg['y_pred_ls'] = y_pred_ls
        print(Lassoreg['y_pred_ls'])

        #prediction dataframe of rr
        Ridgereg = pd.DataFrame(y_test['y_test'])
        Ridgereg['y_pred_rr'] = y_pred_rr
        print(Ridgereg['y_pred_rr'])

        
        #check high r2score and taken final model
        r2_score_lst = []
        r2_score_lst.append(score_lr)
        r2_score_lst.append(score_ls)
        r2_score_lst.append(score_rr)

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        return r2_score_lst, Linearreg, Lassoreg, Ridgereg, regressor, ls, rr, x_test, y_test
        

