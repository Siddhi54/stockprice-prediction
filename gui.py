import matplotlib.pyplot as plt
from bse_scrap import bse_scrape
import streamlit as st
from PIL import Image
#import our finalise model
import pickle



#call class
bseScrap = bse_scrape()
Company_list = bseScrap.get_security_code()

st.title("Stock Price Prediction of BSE")
image = Image.open('bse.jpg')

st.image(image)

user_input = st.selectbox("Select company", Company_list )
seurity_code = bseScrap.get_securityCode_by_company(user_input)

data = bseScrap.get_databysecurity(seurity_code)

st.subheader('Data from 2022-2023')
st.write(data.describe())

#plot graph closing price
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,7))
plt.plot(data.Close,'red')
st.pyplot(fig)


#plot graph closing price with moving avg of 10days
st.subheader('Closing Price vs Time Chart with 10Days Moving Avg')
fig = plt.figure(figsize = (12,7))
plt.plot(data.ema,'black')
plt.plot(data.Close,'red')
st.pyplot(fig)

r2_score_lst, Linearreg, Lassoreg, Ridgereg, regressor, ls, rr, x_test, y_test = bseScrap.train_test(data)

r2_score_maxindex = r2_score_lst.index(max(r2_score_lst))
result = r2_score_maxindex + 1

if result == 1:
    pipe = pickle.load(open("Linearmodel.pkl", 'rb'))
    predictor = pipe.predict(x_test)
    print('pred',predictor)
    pred = predictor.mean()
    #plot graph
    st.subheader('Prediction vs Actual')
    fig =Linearreg.plot(figsize = (12,8), y=['y_test','y_pred_lr'], label =['Original price', 'Predicted price(lr)'])
    plt.ylabel('Price')
    plt.show()
    g = plt.savefig('op.png')
    #for error
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(g)
    #show predicted price
    st.subheader('Predicted Value')
    if st.button('Click here'):
        st.write('Predicted Value is:  ', str(pred))
    
   
if result == 2:
    pipe = pickle.load(open("Lassomodel.pkl", 'rb'))
    predictor = pipe.predict(x_test)
    print('pred',predictor)
    pred = predictor.mean()
    #plot graph
    st.subheader('Prediction vs Actual')
    fig = Lassoreg.plot(figsize = (12,8),y=['y_test','y_pred_ls'], label =['Original price', 'Predicted price(ls)'])
    plt.ylabel('Price')
    plt.show()
    g = plt.savefig('op.png')
    #for error
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(g)
    #show predicted price
    st.subheader('Predicted Value')
    if st.button('Click here'):
        st.write('Predicted Value is:  ', str(pred))
    
    

if result == 3:
    pipe = pickle.load(open("Ridgemodel.pkl", 'rb'))
    predictor = pipe.predict(x_test)
    print('pred',predictor)
    pred = predictor.mean()
    #plot graph
    st.subheader('Prediction vs Actual')
    fig =Lassoreg.plot(figsize = (12,8),y=['y_test','y_pred_rr'], label =['Original price', 'Predicted price(rr)'])
    plt.ylabel('Price')
    plt.show()
    g = plt.savefig('op.png')
    st.pyplot(g)
    #for error
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #show predicted price
    st.subheader('Predicted Value')
    if st.button('Click here'):
        st.write('Predicted Value is:  ', str(pred))

    print("Completed")

