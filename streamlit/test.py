import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


st.title("Streamlit 예제")
st.write("안녕하세요! 이것은 Streamlit 예제입니다.")

st.markdown("""
여기부터 *마크다운* 텍스트입니다.
# 타이틀
```json
{
    'name': 'hello'
}
```

> 여기까지
""")
st.write("이것은 <i>HTML</i> 텍스트입니다.<br/> <a href='javascript:alert(\"ok\");'>click</a>", unsafe_allow_html=True)

# 라인 차트
st.title("라인 차트 예제")
df = pd.DataFrame({
    'x': np.arange(10),
    'y': np.random.randn(10)
})
st.line_chart(df)

df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})
st.line_chart(df, use_container_width=True)

# 바 차트
st.title("바 차트 예제")
df = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Value': [3, 7, 2, 5]
})
st.bar_chart(df.set_index('Category'))

# Altair를 사용한 차트
st.title("Altair 차트 예제")
import altair as alt
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})
c = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
)
st.altair_chart(c, use_container_width=True)

user_input = st.text_input("사용자 이름을 입력하세요", "기본값")
st.write("입력한 사용자 이름:", user_input)

value = st.slider("슬라이드를 움직여보세요", min_value=0, max_value=100, value=50)
st.write(f"선택된 값: {value}")

option = st.radio("라디오 버튼을 선택하세요", ["옵션 1", "옵션 2", "옵션 3"])
st.write(f"선택된 옵션: {option}")

option = st.checkbox("옵션 선택")
if option:
    st.write("옵션이 선택되었습니다!")
else:
    st.write("옵션이 선택 해제되었습니다.")

option = st.selectbox("옵션을 선택하세요", ["옵션 1", "옵션 2", "옵션 3"])
st.write(f"선택된 옵션: {option}")

options = st.multiselect("여러 옵션을 선택하세요", ["옵션 1", "옵션 2", "옵션 3"])
st.write(f"선택된 옵션: {options}")


st.title("대시보드 예제")

page = st.sidebar.selectbox("페이지 선택", ["대시보드", "설정", "도움말"])

if page == "대시보드":
    st.header("대시보드 페이지")
    # 페이지 내용 추가
elif page == "설정":
    st.header("설정 페이지")
    # 페이지 내용 추가
else:
    st.header("도움말 페이지")
    # 페이지 내용 추가

# 파일 업로드
uploaded_image = st.file_uploader("이미지 파일을 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # 이미지 표시
    st.image(uploaded_image, caption="업로드된 이미지", use_column_width=True)

    # 이미지 저장
    with open('uploaded_image.jpg', 'wb') as f:
        f.write(uploaded_image.read())

    st.write("이미지 업로드 및 저장 완료.")
    
if st.button("클릭하세요"):
    st.write("버튼이 클릭되었습니다!")

    
try:
    # 오류가 발생할 수 있는 코드
    result = 1 / 0
except ZeroDivisionError as e:
    st.error(f"오류 발생: {e}")
    