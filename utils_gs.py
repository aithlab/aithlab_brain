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

import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials

@st.cache_data
def get_document(credentials, document_id):
    _credentials = Credentials.from_service_account_info(credentials)
    service = build('drive', 'v3', credentials=_credentials)

    # 파일 내용 요청 및 다운로드
    request = service.files().get_media(fileId=document_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    # 파일 내용을 읽어옵니다.
    fh.seek(0)
    content = fh.read().decode('utf-8')

    savedir_tmp = "./.cache/files/"
    os.makedirs(savedir_tmp, exist_ok=True)
    file_path = os.path.join(savedir_tmp, 'temp.txt')
    with open(file_path, "w") as f:
        f.write(content)
    return file_path