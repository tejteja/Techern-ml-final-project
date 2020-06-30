import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
df=pd.read_csv("placements-df.csv")

###################################label encoding################################################
le=LabelEncoder()
#gender:male=1 female=0
df["gender"]=le.fit_transform(df["gender"])
#10th board: central=0 others=1
df["10th board"]=le.fit_transform(df["10th board"])
#12th board central=0  others=1
df["12th board"]=le.fit_transform(df["12th board"])
#ug stram commerece=1  arts=0   science=2
df["ug stram"]=le.fit_transform(df["ug stram"])
#workex  y=1  n=0
df["workex"]=le.fit_transform(df["workex"])
#degree_t  sci&tech=2 comm&mng=0  others=1
df["degree_t"]=le.fit_transform(df["degree_t"])
#mba specialisation mkt&fin=0    mkt&hr=1
df["specialisation"]=le.fit_transform(df["specialisation"])
#status p=1  np=0
df["status"]=le.fit_transform(df["status"])
################################# dealing null values############################################
df["salary"]=df["salary"].replace(np.NaN,0)
df.head()
#################################### train test split##########33################################
x=df[[ 'gender', '10th grade', '10th board', '12th grade','12th board', 'ug stram', 'ug grade', 'degree_t', 'workex', 'etest_p','specialisation', 'mba_p']]
y1=df["status"]
y2=df["salary"]

##############################logistic regression################################################
lr1=LogisticRegression(C=0.01,solver='liblinear')
lr2=LinearRegression()
lr1.fit(x,y1)
lr2.fit(x,y2)
##############################user data-predections##############################################
print("please enter the required data to predect the possibility of your pacement or to predict the salary")
l=[]
g=int(input("gender:male=1 female=0 \n"))
l.append(g)
a10thg=float(input("10th percent eg:97.0\n"))
l.append(a10thg)
a10b=int(input("10th board: central=0 others=1\n"))
l.append(a10b)
a12g=float(input("12th percent eg:97.0\n"))
l.append(a12g)
a12b=int(input("12th board central=0  others=1\n"))
l.append(a12b)
ugs=int(input("ug stram commerece=1  arts=0   science=2\n"))
l.append(ugs)
ugg=float(input("ug percent eg:97.0\n"))
l.append(ugg)
dt=int(input("degree_t  sci&tech=2 comm&mng=0  others=1\n"))
l.append(dt)
wx=int(input("workex  y=1  n=0\n"))
l.append(wx)
ep=float(input("e_test_p percent eg:97.0\n"))
l.append(ep)
sp=int(input("mba specialisation mkt&fin=0    mkt&hr=1\n"))
l.append(sp)
mp=float(input("mba percent eg:97.0\n"))
l.append(mp)
l=np.array(l).reshape(1,-1)
kn=int(input("enter \n1-to predict the posiibility of placement \n2-to predict salary\n"))
if kn==1:
    ans1=lr1.predict(l)
    sc1=lr1.score(x,y1)*100
    if ans1[0]==1:
        print("can be placed")
    else:
        print("can't be placed ")
elif kn==2:
    ans2=lr2.predict(l)
    sc2=lr2.score(x,y2)*100
    if ans2[0]>0:
        print("salary will be:",ans2[0])
    else:
        print("sorry! salary cant be caliculated...\neaither not placed or details didn't meet requirements...")








