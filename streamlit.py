import streamlit as st
import pickle 
import numpy as np

pickle_in=open('auto_pred.pkl','rb')
auto_Rf=pickle.load(pickle_in)

pickle_in_mlr=open('auto_pred_mlr.pkl','rb')
auto_mlr=pickle.load(pickle_in_mlr)

selected=st.sidebar.radio("Main Menu",['About','Predict price','Model Summary']) 

if selected=='About':
    st.title("About")
    st.write("In this era of technology development, every bsiness wants to make better decisions using data.")
    st.write("In the field of automobiles, data can be used to predict the price of a car based on its specifications.")

elif selected=='Predict price':
    st.title('Predict car price')
    
    doc_file=st.file_uploader("Upload car features",
                             type=['txt'])
    x=[]
    if doc_file is not None:
        lines=doc_file.readlines()
        for l in lines:
            x.append(float(l))
        x=np.array(x).reshape(1,-1)
        
    price,p2=0,0
    if st.button("Predict"):
        #st.write(x[0],x[1])
        price=auto_Rf.predict(x)[0]
        p2=auto_mlr.predict(x)[0]
        
        st.write("\n\n")
        st.write("Predicted price of car\n")
        st.write("Using RandomForestRegressor    : ","%.2f"%price)
        st.write("Using MultipleLinearRegression : ","%.2f"%p2)
        
    
    
elif selected=='Model Summary':
    st.title('Model explained using SHAP')
    
    st.write("Global interpretability")
    st.image('global_bar.jpg')
    st.write("\n\n")
    
    st.image('global_beeswarm.png')
    st.write("\n\n")
    
    st.write('Local interpretability')
    st.image('local_bar.jpg')
    st.write('\n\n')