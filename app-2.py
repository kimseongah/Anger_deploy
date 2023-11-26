import streamlit as st
import torch
import torch.nn as nn
import pandas as pd

import pickle

with open('binary_encoder.pkl', 'wb') as f:
    pickle.dump(binary_encoder, f)
with open('standard_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('binary_encoder.pkl', 'rb') as f:
    loaded_binary_encoder = pickle.load(f)
with open('standard_scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# 전체 모델 불러오기
loaded_model = torch.load('anger_model.pth').to(device)
loaded_model.eval()

# Streamlit 앱 시작
st.title('Anger Prediction App')

# 사용자 입력 받기
name = st.text_input('이름:')
prior_state = st.slider('사건 이전 상태:', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
intensity = st.slider('분노 수준 - 강도:', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
duration = st.slider('분노 수준 - 지속 기간 (분):', min_value=0, max_value=60, value=30, step=1)
expression_method = st.checkbox('표현 방식:')
total_anger_index = st.number_input('총합 분노 지표:', min_value=0, value=20)

# 모델 입력을 위한 데이터 프레임 생성

def decimal_to_binary_list(number):
    binary_string = bin(number)[2:].zfill(5)
    binary_list = [int(bit) for bit in binary_string]
    return tuple(binary_list)

new_data = {
    '이름': ["default"]*15,
    '사건 이전 상태': [prior_state]*15,
    '분노 수준 - 강도': [intensity]*15,
    '분노 수준 - 지속 기간 (분)': [duration]*15,
    '표현 방식': [expression_method]*15,
    '총합 분노 지표': [total_anger_index]*15,
    '선택한 방법': ['default']*15,
    '변화된 분노 척도': [0]*15
}

new_df = pd.DataFrame(new_data)

# 표현 방식 및 대처 방법에 대해 binary_encoder 적용
new_df = loaded_binary_encoder.transform(new_df)

for i in range(1,16):
  a,b,c,d,e = decimal_to_binary_list(i)
  new_df['선택한 방법_0'][i] = a
  new_df['선택한 방법_1'][i] = b
  new_df['선택한 방법_2'][i] = c
  new_df['선택한 방법_3'][i] = d
  new_df['선택한 방법_4'][i] = e

new_input_df = new_df.copy(deep=True)
numeric_index = ['사건 이전 상태', '분노 수준 - 강도', '분노 수준 - 지속 기간 (분)','총합 분노 지표','표현 방식_0', '표현 방식_1', '선택한 방법_0',	'선택한 방법_1',	'선택한 방법_2',	'선택한 방법_3',	'선택한 방법_4']
numeric_features = new_df[numeric_index]

# Standard Scaling
scaled_features = loaded_scaler.transform(numeric_features)

# DataFrame 업데이트
new_df[numeric_index] = scaled_features

# 모델 예측
with torch.no_grad():
    model_input = torch.tensor(new_df[['사건 이전 상태', '분노 수준 - 강도', '분노 수준 - 지속 기간 (분)','총합 분노 지표','표현 방식_0', '표현 방식_1', '선택한 방법_0',	'선택한 방법_1',	'선택한 방법_2',	'선택한 방법_3',	'선택한 방법_4']].values.astype(np.float32), dtype=torch.float32, device=device)
    prediction = loaded_model(model_input)

select = np.argmax(prediction.detach().cpu().numpy())
recommended = loaded_binary_encoder.inverse_transform(new_input_df)['선택한 방법'][select]
# 예측 결과 출력
st.subheader('모델 예측 결과:')
st.write(f'예측 분노 척도: {recommended}')