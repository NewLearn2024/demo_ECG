import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import ECGModel

# 분류에 대한 라벨과 설명
id_to_label = {
    0: "Normal",
    1: "Supraventricular ectopic",
    2: "Ventricular ectopic",
    3: "Fusion of ventricular and normal",
    4: "Fusion of paced and normal"
}

# Streamlit 페이지 설정
st.set_page_config(page_title="ECG 신호 분류", layout="wide")

# 모델과 데이터 로드
@st.cache_resource
def load_model():
    model = ECGModel()
    model.load_state_dict(torch.load('data/MIT_BIH.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_data():
    test_df = pd.read_csv('data/mitbih_test.csv', header=None)
    X_test = torch.tensor(test_df.iloc[:, :187].values, dtype=torch.float32)
    X_test = (X_test - X_test.mean()) / X_test.std()
    return X_test, test_df


model = load_model()
X_test, test_df = load_data()

# UI 설정
st.title('ECG 신호 분류')
st.write('이 앱은 1D-CNN 모델을 사용하여 ECG 신호를 분류합니다.\n\n')
st.write('심전도(ECG)는 피부에 부착된 전극을 통해 일정 시간 동안 심장의 전기적 활동을 기록하는 진단 도구입니다.')
st.write('이러한 신호들은 부정맥, 심근경색, 그 외 다양한 심장 이상 진단에 활용됩니다.')
st.image('data/ecg.webp')
st.write('label 0 Normal: 정상 심장 리듬입니다.')
st.write('label 1 Supraventricular ectopic: 심장의 상위 부분에서 시작되는 비정상적인 signal입니다.')
st.write('label 2 Ventricular ectopic: 심장의 하위 부분에서 시작되는 비정상적인 signal입니다.')
st.write('label 3 Fusion of ventricular and normal: 정상적인 심장 박동과 비정상적인 박동이 혼합된 혼합된 signal입니다.')
st.write('label 4 Fusion of paced and normal: 심장 조율기와 정상 심장 박동이 혼합된 signal입니다.')

# 신호 선택 및 시각화
if 'idx' not in st.session_state or st.button('랜덤 ECG 신호 선택'):
    st.session_state['idx'] = np.random.randint(0, len(X_test))
    idx = st.session_state['idx']

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test_df.iloc[idx, :187])
    ax.set_title('ECG Signal')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    st.pyplot(fig)

# 결과 분석
if st.button('분류 결과 확인 및 비교'):
    idx = st.session_state['idx']
    signal = X_test[idx].unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(signal)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities).item()

    st.write(f"모델 예측 label: {predicted_class}")
    st.write(f"모델 예측 확률: {probabilities[0, predicted_class].item():.4f}")
    st.write(f"실제 label: {int(test_df.iloc[idx, 187])}")
