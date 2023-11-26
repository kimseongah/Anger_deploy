import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

device = 'cpu'

with open('binary_encoder.pkl', 'rb') as f:
    loaded_binary_encoder = pickle.load(f)
with open('standard_scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# 전체 모델 불러오기
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # shortcut 연결을 위한 조건
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)  # shortcut 연결
        
        out = self.relu(out)
        return out

class AngerResNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(AngerResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.residual_block1 = ResidualBlock(hidden_size, hidden_size)
        self.residual_block2 = ResidualBlock(hidden_size, hidden_size)
        self.residual_block3 = ResidualBlock(hidden_size, hidden_size)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if len(x.shape) == 1:
          x = x.unsqueeze(0)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.permute(0, 1, 2, 3).repeat(1, 1, 10, 10)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
loaded_model = torch.load('anger_model.pth').to(device)
loaded_model.eval()

# Streamlit 앱 시작
st.title('분노 관리 방법 추천')

# 사용자 입력 받기
name = st.selectbox('이름: ', ['김성아', '오수진', '장효민', '전동근', '정두용'])
prior_state = st.slider('사건 이전 상태:', min_value=0, max_value=10, value=5, step=1)
intensity = st.slider('분노 수준 - 강도:', min_value=0, max_value=10, value=5, step=1)
duration = st.number_input('분노 수준 - 지속 기간 (분):', min_value=0, max_value=120, value=60, step=5)
expression_method = st.selectbox('표현 방식:', ['억제', '표출', '통제'])
total_anger_index = st.slider('총합 분노 지표:', min_value=0, max_value=10, value=5, step=1)

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
st.subheader('모델 예측 결과: '+recommended)