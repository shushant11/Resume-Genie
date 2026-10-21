import streamlit as st
from PIL import Image

logo= Image.open("logo.jpg")
st.sidebar.image(logo,width=140)

st.sidebar.markdown("**HelloWorldApp**")

st.title("Hello World!")
st.write("Welcome to my first Stremlit application")