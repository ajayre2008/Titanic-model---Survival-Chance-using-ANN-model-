import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model  
import pickle


st.title("passenger Survival Chance in the titanic journey")
pclass=st.slider("Enter the passenger class",1,3)
sex=st.selectbox("choose the gender",["male","female"]  )
sibsp=st.slider("Enter the number of siblings/spouses aboard",0,8)
parch=st.slider("Enter the number of parents/children aboard",0,6)
fare=st.number_input("Enter the fare of the passenger")
embarked=st.selectbox("choose the port of station",["C","Q","S"]  )


data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])

model=load_model('15jun26_ANNmodel.h5')

with open('15jun26_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)  
with open('15jun26_One_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)    
with open('15jun26_Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f) 

data['Sex'] = label_encoder.transform(data['Sex'])
embarked_encoded = one_hot_encoder.transform(data[['Embarked']]).toarray()
embarked_df = pd.DataFrame(embarked_encoded, columns=one_hot_encoder.get_feature_names_out(['Embarked']))
data=pd.concat([data.drop('Embarked', axis=1), embarked_df], axis=1)    


num_cols=['Pclass','SibSp','Parch','Fare']
data[num_cols] = scaler.transform(data[num_cols])
st.write(data) 
y=model.predict(data)

Y=y[0][0]
def chance(y):
    if y>0.5 :
        st.write('The person is likely to survive')
    else:
        st.write('The person is likely to not survive')
if st.button("Predict Survival Chance"):
    st.write("The predicted survival chance is:",Y)
    st.write(chance(Y))

# import streamlit as st
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import pickle

# st.title("Passenger Survival Chance in the Titanic Journey")

# # Inputs
# pclass = st.slider("Enter the passenger class", 1, 3)
# sex = st.selectbox("Choose the gender", ["male", "female"])
# sibsp = st.slider("Enter number of siblings/spouses aboard", 0, 8)
# parch = st.slider("Enter number of parents/children aboard", 0, 6)
# fare = st.number_input("Enter the fare of passenger")
# embarked = st.selectbox("Choose port of station", ["C", "Q", "S"])

# # Load model and preprocessors
# model = load_model("15jun26_ANNmodel.h5")

# with open("15jun26_label_encoder.pkl", "rb") as f:
#     label_encoder = pickle.load(f)

# with open("15jun26_One_hot_encoder.pkl", "rb") as f:
#     one_hot_encoder = pickle.load(f)

# with open("15jun26_Scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)


# if st.button("Predict Survival Chance"):

#     data = pd.DataFrame([{
#         'Pclass': pclass,
#         'Sex': sex,
#         'SibSp': sibsp,
#         'Parch': parch,
#         'Fare': fare,
#         'Embarked': embarked
#     }])

#     # Label encode
#     data["Sex"] = label_encoder.transform(data["Sex"])

#     # One hot encode
#     embarked_encoded = one_hot_encoder.transform(
#         data[["Embarked"]]
#     ).toarray()

#     embarked_df = pd.DataFrame(
#         embarked_encoded,
#         columns=one_hot_encoder.get_feature_names_out(["Embarked"])
#     )

#     data = pd.concat(
#         [data.drop("Embarked", axis=1), embarked_df],
#         axis=1
#     )

#     # Scale full dataset (important)
#     data_scaled = scaler.transform(data)

#     # Predict
#     y = model.predict(data_scaled)

#     probability = float(y[0][0])

#     st.write(
#         f"Predicted survival probability: {probability:.2%}"
#     )

#     if probability > 0.5:
#         st.success("The person is likely to survive")
#     else:
#         st.error("The person is likely to not survive")

