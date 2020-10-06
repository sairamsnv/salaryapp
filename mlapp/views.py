from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
dataset=pd.read_csv(r'https://raw.githubusercontent.com/sairamsnv/salaryapp/master/Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
#import pixiedust
#import requests
#slr=LinearRegression()
#slr.fit(x_train,y_train)
a=0

def fun(request):
    return render(request,'mynew.html')

def fun1(request):
    #if request.method=='POST':
       # try:
         # inspections = pixiedust.sampleData('https://raw.githubusercontent.com/sairamsnv/mycsv/master/Salary_Data.csv')
         # display(inspections)
        #except:
           # print(sys.exc_info()[0])
    #return render(request,'mydisplay.html')
    if request.method=='POST':
        a=request.POST['exp']
        #request.session['name']=request.POST['exp']
        #name1=request.session['name']
        #print(name)
        x = np.asarray(a, dtype='float64')
        slr=LinearRegression()
        slr.fit(x_train,y_train)
        a=np.reshape(-1,1)
        y_predict=slr.predict([[x]])
    return render(request,'mydisplay.html',{'pre': y_predict})
    



    #return render(request,'mydisplay.html',{'pre': y_predict})

def fun2(request):
   
    plt.scatter(x_train,y_train,color='red')
    slr=LinearRegression()
    slr.fit(x_train,y_train)
    global a
    
    x = np.asarray(a, dtype='float64')
    a=np.reshape(-1,1)
    
    plt.plot(x_train,slr.predict(x_train))
    plt.show()
    return render(request,'pixil.html')

    



