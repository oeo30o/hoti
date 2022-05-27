#챗봇
import json
import os
import streamlit as st #웹 인터페이스
from streamlit_chat import message
from numpy import complex256
from sentence_transformers import SentenceTransformer  #트랜스포머 모델
from sklearn.metrics.pairwise import cosine_similarity  #코사인 유사도
import pandas as pd
#stt/tts
import speech_recognition as sr
from speech_recognition import *
from gtts import gTTS
import time
import clipboard
import keyboard
import pyautogui
from pyautogui import *
import playsound
import pyaudio

#페이지 기본 세팅
st.set_page_config(page_title="호티 HOTI", page_icon="🤖",) #페이지 아이콘 및 이름
st.image("https://ifh.cc/g/xNnht7.png", width=300,) #학교 로고
st.title('호티 HOTI') #제목
st.markdown("호치민시한국국제학교 홈페이지 인공지능 챗봇") #설명

#TTS
def speak(text):
    tts = gTTS(text=text,lang='ko')
    filename = 'voice.mp3' #변환된 음성을 mp3 파일에 저장
    tts.save(filename)
    playsound.playsound(filename) #재생

#STT
def read_voice(): #음성 인식
    r = Recognizer() #인식기
    mic = Microphone() #마이크 객체 불러오기
    with mic as source:
        audio = r.listen(source) #음성 읽어오기
    voice_data = r.recognize_google(audio, language='ko') #구글의 한국어 stt 모델 사용
    return voice_data
    
def typing(value): #키보드 입력
    clipboard.copy(value) #복사
    pyautogui.hotkey('command', 'v') #붙여넣기 (pyautogui에서는 영어만 바로 write 할 수 있음)

#Streamlit 세팅
@st.cache(allow_output_mutation=True)
def cached_model(): #모델
    model = SentenceTransformer('jhgan/ko-sroberta-multitask') #Sentence Transformer 모델
    return model

@st.cache(allow_output_mutation=True)
def get_dataset(): #데이터셋
    df = pd.read_csv('chat_real.csv') #전처리 과정을 거친 데이터셋
    df['embedding'] = df['embedding'].apply(json.loads) 
    return df

model = cached_model()
df = get_dataset()

#Session state
if 'past' not in st.session_state:
    st.session_state['past'] = [] #질문 리스트 생성

if 'generated' not in st.session_state:
    st.session_state['generated'] = [] #답변 리스트 생성

#입력창 생성
with st.form('form', clear_on_submit=True):
    user_input = st.text_input("이곳에 질문을 입력하세요")
    col1, col2 = st.columns([9.6,1]) #버튼 위치 조정
    with col1:
        pass
    with col2:  
        submitted = st.form_submit_button('전송') #전송 버튼

#음성인식 버튼 생성
col1, col2 = st.columns([9.6, 1]) #버튼 위치 조정
with col1:
    pass
with col2:
    btn_clicked = st.button("🎙", key="voice_btn") #음성인식 시작 버튼

if btn_clicked:
    voice = read_voice() 
    time.sleep(1)
    #print(voice)
    typing(voice) #입력창에 텍스트 붙여넣기
    #time.sleep(0.5)
    #submitted = True #음성 인식되면 바로 전송 버튼 눌려짐

#답변 생성 단계
if submitted and user_input: #제출을 누르고 텍스트가 질문과 일치하면
    embedding = model.encode(user_input) #입력 텍스트를 인코딩
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze()) #유사도 판정
    answer = df.loc[df['distance'].idxmax()] #유사도가 가장 높은 것을 답변으로 설정
    
    st.session_state.past.clear() #질문 내역 초기화
    st.session_state.generated.clear() #현재 답변 내역 초기화
    st.session_state.past.insert(0,user_input) #사용자 질문 표시 
    st.session_state.generated.insert(0,answer['챗봇']) #답변 표시

    message(st.session_state['past'][0], is_user=True, key=str(0) + '_user') #질문 전송
    time.sleep(0.3) #딜레이
    message(st.session_state['generated'][0], key=str(0) + '_bot') #답변 전송
    k = st.session_state['generated'][0].index('h') #url 시작 인덱스 찾기
    speak(st.session_state['generated'][0][0:k]) #url 제외하고 말하기

    submitted = False #버튼 상태 리셋
