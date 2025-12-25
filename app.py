import streamlit as st
from frontend.the_page import the_page

if "page" not in st.session_state:
    the_page()