from tkinter import *
import matplotlib.pyplot as plt
import csv
import pandas as pd
import tkinter as tk
from tkinter import Message,Text
import cv2 , os
import shutil
import csv
import numpy as np
from PIL import Image ,ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



window=tk.Tk()
window.title("MATHS ASSIGNMENT")
window.geometry('1280x720')
window.configure(background='cyan')
window.grid_rowconfigure(0,weight=1)
window.grid_columnconfigure(0,weight=1)
lb1=tk.Label(window,text="Central Tendency and Dispersion",width=25,height=2,bg='white',fg='black' ,font=('times',15,' bold'))
lb1.place(x=10,y=380)
message2=tk.Label(window,padx=0.5,text='',bg='white',fg='red',width=80,height=2,font=('times',15,'bold '))
message2.place(x=350,y=380)
lb2=tk.Label(window,text="Prediction",width=25,height=2,bg='white',fg='black' ,font=('times',15,' bold'))
lb2.place(x=10,y=450)
message3=tk.Label(window,padx=0.5,text='',bg='white',fg='red',width=80,height=2,font=('times',15,'bold '))
message3.place(x=350,y=450)




def totalcases():
    def prediction():
        df=pd.read_csv('COVID.csv')
        df=df.loc[df['Region'].isin([dropdown])]
        df =df[['Region','Confirmed_Cases','Date']]
        df=df.reset_index()
        df['index']=df.index
        c=df.index[-1]+1
        print(df.index)
        print(df)
        x=np.array(df['index']).reshape(-1,1)
        y=np.array(df['Confirmed_Cases']).reshape(-1,1)
        poly=PolynomialFeatures(degree=3)
        x=poly.fit_transform(x)
        lin_reg=LinearRegression()
        lin_reg.fit(x,y)
        print(lin_reg.score(x,y))
        #plt.plot(df['Confirmed_Cases'])
        #plt.plot(lin_reg.predict(x),'r--')
        print(lin_reg.predict(poly.fit_transform([[c]])))
        #plt.show()
        print(df)
        t=lin_reg.predict(poly.fit_transform([[c]]))
        t=t[0][0]
        return int(t)
    #-----------
    data=pd.read_csv('COVID.csv')
    india=data[data.Region==dropdown]
    me=india.Confirmed_Cases.mean()
    med=india.Confirmed_Cases.median()
    mod=india.Confirmed_Cases.mode()
    dev=india.Confirmed_Cases.std()
    mod=3*med-2*me
    print(me,mod,med)
    me=str(me)
    me=me[0:10]
    med=str(med)
    med=med[0:10]
    #mod=str(mod)
    #mod=mod[1:10]
    c=str('Mean : '+str(me)+'      Median : '+str(med)+'      Standard Deviation : '+str(dev))
    print(c)
    message2.configure(text=c)

    message3.configure(text='Next Day Total Cases will be '+str(prediction()))
    print(india)
    print(c)
    plt.plot(india.Date,india.Confirmed_Cases/10**6,label='Total Cases',color='blue')
    plt.xlabel('Date')
    plt.ylabel('Total no. of Cases(in Millions)')
    plt.title('Covid 19 India('+str(dropdown)+')' )
    plt.legend()
    plt.show()
    

def totaldeath():
    def prediction():
        df=pd.read_csv('COVID.csv')
        df=df.loc[df['Region'].isin([dropdown])]
        df =df[['Region','Death','Date']]
        df=df.reset_index()
        df['index']=df.index
        c=df.index[-1]+1
        print(df.index)
        print(df)
        x=np.array(df['index']).reshape(-1,1)
        y=np.array(df['Death']).reshape(-1,1)
        poly=PolynomialFeatures(degree=3)
        x=poly.fit_transform(x)
        lin_reg=LinearRegression()
        lin_reg.fit(x,y)
        print(lin_reg.score(x,y))
        #plt.plot(df['Confirmed_Cases'])
        #plt.plot(lin_reg.predict(x),'r--')
        print(lin_reg.predict(poly.fit_transform([[c]])))
        #plt.show()
        print(df)
        t=lin_reg.predict(poly.fit_transform([[c]]))
        t=t[0][0]
        return int(t)
    #-----------
    data=pd.read_csv('COVID.csv')
    india=data[data.Region==dropdown]
    me=india.Death_Rate.mean()
    med=india.Death_Rate.median()
    mod=india.Death_Rate.mode()
    dev=india.Death_Rate.std()
    mod=3*med-2*me
    me=str(me)
    me=me[0:10]
    med=str(med)
    med=med[0:10]
    #mod=str(mod)
    #mod=mod[2:7]
    c=str('Mean : '+str(me)+'      Median : '+str(med)+'      Standard Deviation : '+str(dev))
    message2.configure(text=c)
    message3.configure(text='Next Day Total Deaths will be '+str(prediction()))
    print(india)
    plt.plot(india.Date,india.Death_Rate,label='Deaths ',color='red')
    plt.xlabel('date')
    plt.ylabel('Total no. of Deaths Per Day')
    plt.title('Covid 19 India('+str(dropdown)+')' )
    plt.legend()
    plt.show()
def death_rate():
    totaldeath()
    
def total_cases():
    totalcases()


def growthrate():
    def prediction():
        df=pd.read_csv('COVID.csv')
        df=df.loc[df['Region'].isin([dropdown])]
        df =df[['Region','Growth_Rate','Date']]
        df=df.reset_index()
        df['index']=df.index
        c=df.index[-1]+1
        print(df.index)
        print(df)
        x=np.array(df['index']).reshape(-1,1)
        y=np.array(df['Growth_Rate']).reshape(-1,1)
        poly=PolynomialFeatures(degree=3)
        x=poly.fit_transform(x)
        lin_reg=LinearRegression()
        lin_reg.fit(x,y)
        print(lin_reg.score(x,y))
        #plt.plot(df['Confirmed_Cases'])
        #plt.plot(lin_reg.predict(x),'r--')
        print(lin_reg.predict(poly.fit_transform([[c]])))
        #plt.show()
        print(df)
        t=lin_reg.predict(poly.fit_transform([[c]]))
        t=t[0][0]
        return int(t)
    data=pd.read_csv('COVID.csv')
    india=data[data.Region==dropdown]
    me=india.Growth_Rate.mean()
    med=india.Growth_Rate.median()
    mod=india.Growth_Rate.mode()
    dev=india.Growth_Rate.std()
    mod=3*med-2*me
    me=str(me)
    me=me[0:10]
    med=str(med)
    med=med[0:10]
    #mod=str(mod)
    #mod=mod[2:10]
    c=str('Mean : '+str(me)+'      Median : '+str(med)+'      Standard Deviation : '+str(dev))
    message2.configure(text=c)
    message3.configure(text='Next Day Increase in active cases will be '+str(prediction()))
    print(india)
    plt.plot(india.Date,india.Growth_Rate,label='Growth In Active Count',color='grey')
    plt.xlabel('date')
    plt.ylabel('Total no. Cases Increase Per Day')
    plt.title('Covid 19 India('+str(dropdown)+')' )
    plt.legend()
    plt.show()
    
def growth_rate():
    growthrate()

def Recoveries():
    def prediction():
        df=pd.read_csv('COVID.csv')
        df=df.loc[df['Region'].isin([dropdown])]
        df =df[['Region','Cured','Date']]
        df=df.reset_index()
        df['index']=df.index
        c=df.index[-1]+1
        print(df.index)
        print(df)
        x=np.array(df['index']).reshape(-1,1)
        y=np.array(df['Cured']).reshape(-1,1)
        poly=PolynomialFeatures(degree=3)
        x=poly.fit_transform(x)
        lin_reg=LinearRegression()
        lin_reg.fit(x,y)
        print(lin_reg.score(x,y))
        #plt.plot(df['Confirmed_Cases'])
        #plt.plot(lin_reg.predict(x),'r--')
        print(lin_reg.predict(poly.fit_transform([[c]])))
        #plt.show()
        print(df)
        t=lin_reg.predict(poly.fit_transform([[c]]))
        t=t[0][0]
        return int(t)
    #-----------
    data=pd.read_csv('COVID.csv')
    india=data[data.Region==dropdown]
    print(india.Cured.mean())
    me=india.Cured.mean()
    med=india.Cured.median()
    mod=india.Cured.mode()
    dev=india.Cured.std()
    mod=3*med-2*me
    med=str(med)
    med=med[0:10]
    #mod=str(mod)
    #mod=mod[2:6]
    c=str('Mean : '+str(me)+'      Median : '+str(med)+'      Standard Deviation : '+str(dev))
    print(c)
    message2.configure(text=c)
    message3.configure(text='Next Day Total Recoveries will be '+str(prediction()))
    print(india)
    plt.plot(india.Date,india.Cured,label='Total Recoveries',color='green')
    plt.xlabel('Date')
    plt.ylabel('Total no. of Recoveries')
    plt.title('Covid 19 India' )
    plt.legend()
    plt.show()

def recoveries():
    Recoveries()

def averagepermonth():
    data=pd.read_csv('COVID1.csv')
    c=data[dropdown]
    print(c)
    plt.bar(data.Month,c,label='Average Per Month',color='blue')
    plt.xlabel('Date')
    plt.ylabel('Average No. of Cases Per Month')
    plt.title('Covid 19 India('+str(dropdown)+')' )
    plt.legend()
    plt.show()
    

tkvar = StringVar(window)

# Dictionary with options
choices = sorted({'India','Tamil Nadu','Sikkim','Uttarakhand','Tripura','Telangana','Andaman and Nicobar Islands','Arunachal Pradesh','Assam','Haryana','Andhra Pradesh','Goa','Delhi','Chhattisgarh','Maharashtra','--Select--','Himachal Pradesh','Ladakh','Jharkhand','Jammu and Kashmir','West Bengal','Manipur','Bihar','Uttar Pradesh','Puducherry','Odisha', 'Nagaland','Meghalaya','Mizoram','Madhya Pradesh', 'Karnataka', 'Punjab','Kerala','Rajashthan'})
tkvar.set('--Select--')  # set the default option

popupMenu = OptionMenu(window, tkvar, *choices)
popupMenu.grid(row=7, column=2)
popupMenu.config(width=20,height=2)
popupMenu.place(x=970,y=140)


def change_dropdown(*args):
    global dropdown
    dropdown = str(tkvar.get())
    print(dropdown)
    if tkvar.get() == 'India':
        pass

tkvar.trace('w', change_dropdown)

message=tk.Label(window,padx=0.5,text='DATA ANALYSIS OF COVID 19 INDIA',bg='cyan',fg='goldenrod4',width=50,height=4,font=('times',30,'bold '))
message.place(x=100,y=-40)

clearButton=tk.Button(window,text='Total Cases', command=total_cases,bg='grey',fg='lightyellow',width=10,height=1, activebackground='red',font=('times',30,'bold'))
clearButton.place(x=50,y=140)

clearButton=tk.Button(window,text='Growth Rate', command=growth_rate,bg='grey',fg='lightyellow',width=10,height=1, activebackground='red',font=('times',30,'bold'))
clearButton.place(x=350,y=140)

clearButton=tk.Button(window,text='Average Per Month', command=averagepermonth,bg='grey',fg='lightyellow',width=15,height=1, activebackground='red',font=('times',30,'bold'))
clearButton.place(x=50,y=250)

clearButton=tk.Button(window,text='Death Rate', command=death_rate,bg='grey',fg='lightyellow',width=10,height=1, activebackground='red',font=('times',30,'bold'))
clearButton.place(x=650,y=140)

clearButton=tk.Button(window,text='Total Recoveries', command=recoveries,bg='grey',fg='lightyellow',width=15,height=1, activebackground='red',font=('times',30,'bold'))
clearButton.place(x=550,y=250)

quitwindow=tk.Button(window,text='Quit',command=window.destroy,bg='red3',fg='lightyellow',width=10,height=1, activebackground='white',font=('times',30,'bold'))
quitwindow.place(x=1050,y=540)











window.mainloop()
