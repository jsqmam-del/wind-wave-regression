"""선박 데이터 회귀분석 프로그램 (데이터처리개론 과제 1)
실행: streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import regression as rg

st.set_page_config(page_title="해양 관측 데이터 회귀분석", page_icon="🌊", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
SAMPLES = sorted(p for p in DATA_DIR.glob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls", ".data"))
POINT = "#1f5f8b"
LINE = "#e07a1f"

st.title("🌊 해양 관측 데이터 회귀분석 프로그램")
st.caption("데이터를 불러오고 X/Y를 골라 선형회귀와 비선형회귀를 비교합니다. "
           "회귀 알고리즘은 라이브러리 없이 numpy로 직접 구현했습니다.")

# ================================================================ 사이드바: 데이터
with st.sidebar:
    st.header("1. 데이터 불러오기")
    uploaded = st.file_uploader("CSV, Excel, 또는 공백 구분 텍스트", type=["csv", "xlsx", "xls", "data", "txt"])
    sample = None
    if uploaded is None and SAMPLES:
        sample = st.selectbox("또는 저장된 데이터 사용", SAMPLES, format_func=lambda p: p.name)

df = None
try:
    if uploaded is not None:
        df = rg.load_table(uploaded.name, uploaded.getvalue())
    elif sample is not None:
        df = rg.load_table(sample.name, sample.read_bytes())
except Exception as e:
    st.error(f"파일을 읽지 못했습니다: {e}. 첫 줄이 열 이름인 CSV인지 확인하세요.")

if df is None:
    st.info("왼쪽에서 데이터 파일을 올리면 분석을 시작합니다.")
    st.stop()

num_cols = rg.numeric_columns(df)
if len(num_cols) < 2:
    st.error("숫자로 된 열이 2개 이상 있어야 회귀분석을 할 수 있습니다.")
    st.stop()

with st.sidebar:
    st.header("2. 축 선택")
    def find(keywords, fallback):
        for kw in keywords:
            for i, c in enumerate(num_cols):
                if kw in str(c) and "GUST" not in str(c).upper():
                    return i
        return fallback
    default_x = find(["풍속", "프루드수"], 0)
    default_y = find(["유의파고", "잉여저항"], min(1, len(num_cols) - 1))
    x_col = st.selectbox("X축 (입력 변수)", num_cols, index=default_x)
    y_col = st.selectbox("Y축 (예측할 값)", num_cols, index=default_y)

    st.header("3. 데이터 정제")
    drop_neg = st.checkbox("0 이하 값 제거 (결측·오류값)", value=True,
                           help="관측 장비 오류나 결측이 음수나 0으로 기록된 경우를 제외합니다.")

if x_col == y_col:
    st.warning("X축과 Y축에 서로 다른 열을 골라주세요.")
    st.stop()

data = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
n_raw = len(data)
if drop_neg:
    data = data[(data[x_col] > 0) & (data[y_col] > 0)]
with st.sidebar:
    st.caption(f"분석에 사용하는 데이터: {len(data):,}개 (제외 {n_raw - len(data):,}개)")
if len(data) < 5:
    st.error("분석할 수 있는 데이터가 너무 적습니다. 다른 열을 고르거나 정제 옵션을 꺼보세요.")
    st.stop()
x = data[x_col].to_numpy(dtype=float)
y = data[y_col].to_numpy(dtype=float)


def scatter(title=""):
    big = len(x) > 2000  # 1년치 시간자료(약 8,760개)도 부드럽게 그리도록 WebGL 사용
    fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers", name="데이터",
                                 marker=dict(color=POINT, size=4 if big else 7, opacity=0.3 if big else 0.6)))
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col,
                      height=460, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.02, x=0))
    return fig


x_grid = np.linspace(x.min(), x.max(), 300)

tab_data, tab_lin, tab_poly, tab_nn = st.tabs(["데이터", "선형회귀", "비선형회귀", "인공신경망 (다음 과제)"])

# ================================================================ 탭 1: 데이터
with tab_data:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("데이터 미리보기")
        st.write(f"{len(df)}행 × {df.shape[1]}열")
        st.dataframe(df.head(50), height=360)
    with c2:
        st.subheader(f"{x_col} vs {y_col}")
        st.plotly_chart(scatter())
    st.subheader("기초 통계")
    st.dataframe(df[num_cols].describe().T.round(4))

# ================================================================ 탭 2: 선형회귀
with tab_lin:
    st.markdown("**모델:** y = a·x + b  \n잔차(실제값 − 예측값)의 제곱합이 가장 작아지는 a, b를 찾습니다.")
    c1, c2 = st.columns(2)
    lr = c1.select_slider("학습률 (한 번에 이동하는 거리)", options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 1.1], value=0.1)
    n_iter = c2.slider("반복 횟수", 10, 2000, 300, step=10)

    a_gd, b_gd, losses, snaps = rg.gradient_descent_linear(x, y, lr, n_iter)
    a_ls, b_ls = rg.least_squares_linear(x, y)

    if not np.isfinite(a_gd):
        st.error("학습률이 너무 커서 손실이 발산했습니다. 학습률을 낮춰보세요.")
    else:
        left, right = st.columns(2)
        with left:
            fig = scatter("경사하강법으로 회귀선이 맞춰지는 과정")
            for i, (it, a_s, b_s) in enumerate(snaps):
                fig.add_trace(go.Scatter(x=x_grid, y=a_s * x_grid + b_s, mode="lines",
                                         line=dict(color="gray", width=1, dash="dot"),
                                         opacity=0.3 + 0.5 * i / max(1, len(snaps) - 1),
                                         name=f"{it + 1}회", showlegend=False))
            fig.add_trace(go.Scatter(x=x_grid, y=a_gd * x_grid + b_gd, mode="lines",
                                     line=dict(color=LINE, width=3), name="최종 회귀선"))
            st.plotly_chart(fig)
        with right:
            lf = go.Figure(go.Scatter(y=losses, mode="lines", line=dict(color=POINT)))
            lf.update_layout(title="반복에 따른 손실(평균 잔차 제곱) 감소", xaxis_title="반복 횟수",
                             yaxis_title="손실", yaxis_type="log", height=460,
                             margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(lf)

        y_hat = a_gd * x + b_gd
        st.subheader("결과 비교: 경사하강법 vs 최소제곱법 공식")
        st.dataframe(pd.DataFrame({
            "방법": ["경사하강법 (반복 탐색)", "최소제곱법 (정규방정식)"],
            "기울기 a": [a_gd, a_ls],
            "절편 b": [b_gd, b_ls],
            "R²": [rg.r2_score(y, y_hat), rg.r2_score(y, a_ls * x + b_ls)],
            "RMSE": [rg.rmse(y, y_hat), rg.rmse(y, a_ls * x + b_ls)],
        }).round(5), hide_index=True)
        st.caption("반복 횟수가 충분하면 두 방법의 a, b가 같아집니다. "
                   "학습률을 너무 크게 하면 발산하고, 너무 작게 하면 바닥까지 가지 못합니다.")

        st.subheader("잔차 그래프")
        rf = go.Figure(go.Scattergl(x=x, y=y - y_hat, mode="markers", marker=dict(color=POINT, size=4, opacity=0.4)))
        rf.add_hline(y=0, line_color=LINE)
        rf.update_layout(xaxis_title=x_col, yaxis_title="잔차 (실제 − 예측)", height=320,
                         margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(rf)
        st.caption("잔차가 0 주변에 고르게 흩어지지 않고 곡선 모양을 그리면, 직선이 맞지 않는 데이터라는 뜻입니다 → 비선형회귀 탭으로.")

# ================================================================ 탭 3: 비선형회귀
with tab_poly:
    st.markdown("**모델:** y = c₀ + c₁x + c₂x² + … + c_d x^d  \n"
                "테일러 급수처럼 고차항을 더해 곡선으로 맞춥니다. 1차면 선형회귀와 같습니다.")
    c1, c2 = st.columns(2)
    degree = c1.slider("다항식 차수", 1, 8, 3)
    test_ratio = c2.slider("테스트 데이터 비율", 0.1, 0.4, 0.2, step=0.05)

    model = rg.PolyModel(degree).fit(x, y)
    y_hat = model.predict(x)

    a_pw, b_pw, n_pw = rg.power_fit(x, y)
    y_pw = a_pw * np.clip(x, 1e-12, None) ** b_pw

    left, right = st.columns(2)
    with left:
        fig = scatter(f"{degree}차 다항 회귀")
        bins = rg.bin_means(x, y)
        if len(bins):
            fig.add_trace(go.Scatter(x=bins["구간 중앙"], y=bins["평균"], mode="markers",
                                     marker=dict(color="#c1121f", size=11, symbol="diamond",
                                                 line=dict(color="white", width=1)),
                                     name="구간별 평균"))
        if n_pw >= 2:
            xg = x_grid[x_grid > 0]
            fig.add_trace(go.Scatter(x=xg, y=a_pw * xg ** b_pw, mode="lines",
                                     line=dict(color="#2a9d8f", width=2), name="거듭제곱 y=a·xᵇ"))
        fig.add_trace(go.Scatter(x=x_grid, y=a_ls * x_grid + b_ls, mode="lines",
                                 line=dict(color="gray", dash="dash"), name="선형회귀 (비교)"))
        fig.add_trace(go.Scatter(x=x_grid, y=model.predict(x_grid), mode="lines",
                                 line=dict(color=LINE, width=3), name=f"{degree}차 곡선"))
        st.plotly_chart(fig)
    with right:
        table = rg.degree_table(x, y, max_degree=8, test_ratio=test_ratio)
        tf = go.Figure()
        tf.add_trace(go.Scatter(x=table["차수"], y=table["학습 R²"], mode="lines+markers", name="학습 R²"))
        tf.add_trace(go.Scatter(x=table["차수"], y=table["테스트 R²"], mode="lines+markers", name="테스트 R²"))
        tf.update_layout(title="차수별 성능 (과적합 확인)", xaxis_title="차수", yaxis_title="R²",
                         height=460, margin=dict(l=10, r=10, t=50, b=10),
                         legend=dict(orientation="h", y=1.02, x=0))
        st.plotly_chart(tf)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (선형)", f"{rg.r2_score(y, a_ls * x + b_ls):.4f}")
    m2.metric(f"R² ({degree}차)", f"{rg.r2_score(y, y_hat):.4f}")
    m3.metric("R² (거듭제곱)", f"{rg.r2_score(y, y_pw):.4f}" if n_pw >= 2 else "-")
    m4.metric(f"RMSE ({degree}차)", f"{rg.rmse(y, y_hat):.4f}")

    if n_pw >= 2:
        st.markdown("**거듭제곱 모델** (양변에 로그를 취해 직선으로 바꾼 뒤 최소제곱법으로 계산)")
        st.code(f"y = {a_pw:.4g} · x^{b_pw:.3f}", language=None)
        st.caption(f"지수 b = {b_pw:.2f}. 파랑 이론에서 충분히 발달한 풍파의 유의파고는 풍속의 제곱(b≈2)에 비례합니다. "
                   "거듭제곱 모델은 풍속 0일 때 파고도 0이라고 가정하지만, 실제 바다에는 바람이 없어도 먼 곳에서 온 "
                   "너울이 있어서 이 가정이 맞지 않으면 지수가 작게 나오고 R²도 낮아집니다. "
                   "이 경우 '기본 파고 + 풍속² 항' 형태인 2차 다항식이 더 잘 맞습니다.")
    st.markdown(f"**{degree}차 다항 회귀식**")
    st.code(rg.format_poly(model.coef_original(), "x"), language=None)
    st.dataframe(table.round(4), hide_index=True)
    best = int(table.loc[table["테스트 R²"].idxmax(), "차수"])
    st.caption(f"테스트 R²가 가장 높은 차수는 {best}차입니다. 학습 R²는 차수를 올릴수록 계속 좋아지지만, "
               "테스트 R²가 떨어지기 시작하면 데이터의 잡음까지 외운 과적합입니다.")

# ================================================================ 탭 4: 다음 과제 자리
with tab_nn:
    st.info("다음 과제에서 인공신경망 회귀를 이 탭에 추가할 예정입니다. "
            "다항 회귀는 차수를 사람이 골라야 하지만, 신경망은 곡선의 모양을 데이터로부터 스스로 학습합니다.")
