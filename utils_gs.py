import gspread
import numpy as np
import pandas as pd
import streamlit as st
from gspread_dataframe import set_with_dataframe

def get_df_from_url(url):
    # print('Call get_df_from_url')
    csv_export_url = url.replace('/edit#gid=', '/export?format=csv&gid=')
    df = pd.read_csv(csv_export_url)
    return df

@st.cache_resource
def authorize_with_credentials(credentials):
    # print('Call authorize_with_credentials')
    gc = gspread.service_account_from_dict(credentials)
    return gc

@st.cache_resource
def open_gs_by_url(_gc, url, sheet_name='Sheet1'):
    # print('Call open_gs_by_url')
    gs = _gc.open_by_url(url)
    worksheet = gs.worksheet(sheet_name)
    return worksheet

def save_df_to_gd(df, worksheet):
    worksheet.clear()
    set_with_dataframe(worksheet=worksheet, dataframe=df, 
                       include_index=False, include_column_header=True, 
                       resize=True)

@st.cache_data
def get_session_id(df):
    exists = df['Session Id'] 
    id_candi = np.random.randint(10000000, 99999999)
    while id_candi in exists:
        id_candi = np.random.randint(10000000, 99999999)
    return id_candi