
import streamlit as slt
import streamlit.components.v1 as slcom
import helper
import pickle
import base64
from jinja2 import Template


model = pickle.load(open('model.pkl','rb'))

slt.header('Duplicate Question Pairs')
slt.markdown('---')

q1 = slt.text_input('Question 1','Type First Question')
q2 = slt.text_input('Question 2','Type Second Question')

if slt.button('Check'):
    query = helper.quest(q1,q2)
    result = model.predict(query)[0]

    if result:
        slt.header('Duplicate')
    else:
        slt.header('Not Duplicate')

slt.markdown('---')





