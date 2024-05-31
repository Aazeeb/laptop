import streamlit as st
import pickle
import numpy as np

#Import the model
df=pickle.load(open('df.pkl','rb'))
pipe=pickle.load(open('pipe.pkl','rb'))

st.title("Laptop Price Predictor")
company=st.selectbox("Brand",df['Company'].unique(),index=4)
type=st.selectbox("Type",df['TypeName'].unique(),index=1)
cpu=st.selectbox("Cpu",df['Cpu'].unique(),index=0)
ram=st.selectbox("Ram",[2,4,6,8,12,16,24,32,64,128],index=3)
gpu=st.selectbox("Gpu",df['Gpu'].unique(),index=0)
os=st.selectbox("Os",df['OpSys'].unique(),index=2)
weight=st.number_input("Weight(in kg)",min_value=0.6,max_value=4.7,value=2.0,step=0.1)
touchscreen=st.selectbox("Touchscreen",['Yes','No'])
ips=st.selectbox("ips",['Yes','No'])
screen_size=st.number_input("Screen size(in Inches),calculated diagonally",
min_value=10.0,max_value=18.5,value=15.6,step=0.1)
resolution=st.selectbox("Screen Resolution",['2560x1600', '1440x900', '1920x1080', '2880x1800', '1366x768', '2304x1440',
 '3200x1800', '1920x1200', '2256x1504', '3840x2160', '2160x1440', '2560x1440', '1600x900', '2736x1824', '2400x1600'],index=2)

if st.button("PREDICT"):
  ppi=None
  if(touchscreen=='Yes'):
    touchscreen=1
  else:
    touchscreen=0
  if (ips=='Yes'):
    ips=1
  else:
    ips=0
  X_res=int(resolution.split('x')[0])
  Y_res=int(resolution.split('x')[1])
  ppi=((X_res**2)+(Y_res**2))**0.5/screen_size
  query=np.array([[company,type,cpu,ram,gpu,os,weight,touchscreen,ips,ppi]])
  op=np.exp(pipe.predict(query))
  st.subheader("The predicted price of the laptop for the above configuration is "+str(round(op[0]))+" pounds.")
