import numpy as np
from Newclass import *
import time, datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
my_email = "rhakgkql@naver.com"
pwd = "V66NQVFQMLWU"
to_mail = "momichael98@gmail.com"

lat = [[1.0,0.0,0.0],[1/2,np.sqrt(3)/2,0.0],[0.0,0.0,10.0]]
pos = [[1/3,1/3,1/2],[2/3,2/3,1/2]]
ns = 1
soc = False
rkgrid = [15,15,1]
orboption = [[0,1],[1,1]]
N = 1

func = CorrelationFunction(lat,pos,ns,soc,rkgrid,orboption,N)
t = -2.7
hopplist = [[t,0,1,[0,0,0]],[t,1,0,[1,0,0]],[t,1,0,[0,1,0]]]
e = 0.0
onsitelist = [e,e]
hamtb = func.TighBinding(hopplist,onsitelist)

U = 5
option = {1: {"KorS" : "S", "value" : [U,0,0], "site" : 0, "orbitals" : [0]}, 2: {"KorS" : "S", "value" : [U,0,0], "site" : 0, "orbitals" : [1]}}
V = 2
impamp = [[V,0,1,[0,0,0]],[V,1,0,[1,0,0]],[V,1,0,[0,1,0]]]
iter = 100
mix = 1
T = 300
size = 1000
print("HF start")
start = time.time()
func.HartreeFock(iter,mix,T,size,hopplist,onsitelist,option,impamp)
end = time.time()
print("HF finish")
delta = datetime.timedelta(seconds=(end-start))
print(f"HF loop time : {delta}")
msg = MIMEMultipart()
msg['Subject'] = "HF test"
msg['From'] = my_email
msg['To'] = to_mail

text = MIMEText("HF test finish you have to check the result from your labtop")
msg.attach(text)
smtp = smtplib.SMTP("smtp.naver.com",587)
smtp.starttls()
smtp.login(user=my_email,password=pwd)
smtp.sendmail(my_email,to_mail,msg.as_string())
smtp.close()
hamhf = func.hamhf
cry = Crystal(lat,pos,ns,soc,rkgrid,orboption,N)
temp = FLatStc(cry)
energyhf = temp.Diagonalize(hamhf)
print(energyhf[0,0].max(),energyhf[1,1].min())
temp.Visualization(energyhf,'hf.png')  
#print("GW start")
#start = time.time()
#func.GWApproximation(iter,mix,T,size,hopplist,onsitelist,option,impamp)
#end = time.time()
#print("GW finish")
#delta = datetime.timedelta(seconds=(end-start))
#print(f"HF loop time : {delta}")
#
#hamqp = func.hamqp.hamqp
#energygw = temp.Diagonalize(hamqp)
#print(energygw[0,0].max(),energygw[1,1].min())
#
#
#msg = MIMEMultipart()
#msg['Subject'] = "GW test"
#msg['From'] = my_email
#msg['To'] = to_mail
#
#text = MIMEText("GW test finish you have to check the result from your labtop")
#msg.attach(text)
#smtp = smtplib.SMTP("smtp.naver.com",587)
#smtp.starttls()
#smtp.login(user=my_email,password=pwd)
#smtp.sendmail(my_email,to_mail,msg.as_string())
#smtp.close()
