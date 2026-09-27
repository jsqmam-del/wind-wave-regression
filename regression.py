"""회귀 알고리즘 모듈 (라이브러리 없이 numpy로 직접 구현)

- 최소제곱법(정규방정식)으로 선형회귀 계수 구하기
- 경사하강법(Gradient Descent)으로 선형회귀 계수 구하기
- 다항 회귀(비선형회귀): 테일러 급수처럼 x, x^2, x^3 ... 항을 더해 곡선으로 맞추기
"""
import io
import numpy as np
import pandas as pd

YACHT_COLUMNS = [
    "부력중심위치(LCB)",
    "프리즘계수(Cp)",
    "길이-배수량비",
    "폭-흘수비(B/T)",
    "길이-폭비(L/B)",
    "프루드수(Fn)",
    "잉여저항(Rr)",
]


# ---------------------------------------------------------------- 데이터 불러오기
def load_table(file_name: str, raw: bytes) -> pd.DataFrame:
    """CSV, Excel, 공백 구분 텍스트(.data/.txt)를 모두 DataFrame으로 읽는다."""
    name = file_name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw))
    elif name.endswith(".csv"):
        # 기상청 등 국내 공공데이터 CSV는 대부분 CP949(EUC-KR) 인코딩이라 차례로 시도
        for enc in ("utf-8-sig", "cp949", "euc-kr"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("파일 인코딩을 읽을 수 없습니다 (UTF-8 또는 CP949로 저장해 주세요)")
    else:  # UCI 요트 데이터처럼 공백으로 구분된 헤더 없는 파일
        df = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None)
        if df.shape[1] == len(YACHT_COLUMNS):
            df.columns = YACHT_COLUMNS
        else:
            df.columns = [f"열{i + 1}" for i in range(df.shape[1])]
    return df.dropna(how="all")


def numeric_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------- 평가 지표
def r2_score(y, y_pred) -> float:
    ss_res = np.sum((y - y_pred) ** 2)          # 잔차 제곱합
    ss_tot = np.sum((y - np.mean(y)) ** 2)      # 평균 대비 전체 변동
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def rmse(y, y_pred) -> float:
    return float(np.sqrt(np.mean((y - y_pred) ** 2)))


# ---------------------------------------------------------------- 선형회귀 ①: 최소제곱법
def least_squares_linear(x, y):
    """y = a*x + b 에서 잔차 제곱합을 최소로 만드는 a, b를 공식으로 바로 구한다.
    (잔차 제곱합을 a, b로 각각 미분해서 0이 되는 점 = 정규방정식)"""
    x_mean, y_mean = np.mean(x), np.mean(y)
    a = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - a * x_mean
    return float(a), float(b)


# ---------------------------------------------------------------- 선형회귀 ②: 경사하강법
def gradient_descent_linear(x, y, learning_rate=0.1, n_iter=500, init_a=None, init_b=None, seed=0):
    """경사하강법으로 y = a*x + b 를 찾는다.

    1) a, b를 랜덤하게 하나 찍는다 (수업의 '아무 숫자나 넣어보기')
    2) 손실 L = 평균(잔차^2)의 기울기(미분값)를 구한다
    3) 기울기가 내려가는 방향으로 learning_rate 만큼 조금 이동한다
    4) 반복하면 바가지(2차 함수) 모양 손실의 바닥에 도착한다

    x 값의 크기가 크면 발산하기 쉬워서, 내부적으로 x를 표준화(평균 0, 표준편차 1)해서
    학습한 뒤 마지막에 원래 단위의 a, b로 되돌린다.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, sigma = x.mean(), x.std() or 1.0
    xs = (x - mu) / sigma

    rng = np.random.default_rng(seed)
    w = rng.normal() * 10 if init_a is None else init_a * sigma          # 표준화 공간의 기울기
    c = rng.normal() * 10 if init_b is None else init_b + (init_a or 0) * mu  # 표준화 공간의 절편
    n = len(xs)

    losses, snapshots = [], []
    snap_at = set(np.unique(np.geomspace(1, n_iter, 8).astype(int)) - 1) | {0}
    for i in range(n_iter):
        y_pred = w * xs + c
        residual = y_pred - y
        losses.append(float(np.mean(residual ** 2)))
        if i in snap_at:
            snapshots.append((i, w / sigma, c - w * mu / sigma))
        grad_w = 2 / n * np.sum(residual * xs)   # dL/dw
        grad_c = 2 / n * np.sum(residual)        # dL/dc
        w -= learning_rate * grad_w
        c -= learning_rate * grad_c
        if not np.isfinite(w) or not np.isfinite(c):
            break  # 학습률이 너무 크면 발산

    a = w / sigma
    b = c - w * mu / sigma
    return float(a), float(b), losses, snapshots


# ---------------------------------------------------------------- 비선형회귀: 다항 회귀
class PolyModel:
    """y = c0 + c1*x + c2*x^2 + ... + cd*x^d
    계수에 대해서는 여전히 '선형'이라서 선형회귀와 똑같이 잔차 제곱합 최소화로 풀린다.
    수치 안정성을 위해 x를 표준화한 뒤 최소제곱(lstsq)으로 푼다."""

    def __init__(self, degree: int):
        self.degree = degree

    def _design(self, x):
        z = (np.asarray(x, dtype=float) - self.mu) / self.sigma
        return np.vander(z, self.degree + 1, increasing=True)

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        self.mu, self.sigma = x.mean(), x.std() or 1.0
        self.coef, *_ = np.linalg.lstsq(self._design(x), np.asarray(y, dtype=float), rcond=None)
        return self

    def predict(self, x):
        return self._design(x) @ self.coef

    def coef_original(self):
        """원래 x 단위의 계수 [c0, c1, ..., cd] (보고서에 식을 적을 때 사용)"""
        p = np.polynomial.Polynomial(self.coef)
        shift = np.polynomial.Polynomial([-self.mu / self.sigma, 1 / self.sigma])
        return (p(shift)).coef


def train_test_split(x, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_test = max(1, int(len(x) * test_ratio))
    te, tr = idx[:n_test], idx[n_test:]
    return x[tr], x[te], y[tr], y[te]


def degree_table(x, y, max_degree=8, test_ratio=0.2, seed=42) -> pd.DataFrame:
    """차수별로 학습/테스트 성능을 비교해 과적합 여부를 확인한다."""
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_ratio, seed)
    rows = []
    for d in range(1, max_degree + 1):
        m = PolyModel(d).fit(x_tr, y_tr)
        rows.append({
            "차수": d,
            "학습 R²": r2_score(y_tr, m.predict(x_tr)),
            "테스트 R²": r2_score(y_te, m.predict(x_te)),
            "테스트 RMSE": rmse(y_te, m.predict(x_te)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 비선형회귀: 거듭제곱 모델
def power_fit(x, y):
    """y = a * x^b 를 양변에 로그를 취해 직선으로 바꾼 뒤 최소제곱법으로 푼다.
        ln(y) = ln(a) + b * ln(x)   →   Y = B + b * X  (선형회귀와 같은 문제)
    x, y가 0 이하인 점은 로그를 취할 수 없어서 제외한다."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    b, ln_a = least_squares_linear(np.log(x[mask]), np.log(y[mask]))
    return float(np.exp(ln_a)), float(b), int(mask.sum())


def bin_means(x, y, n_bins=12, min_count=10) -> pd.DataFrame:
    """x를 같은 간격 구간으로 나눠 구간별 y 평균을 구한다. 흩어진 점 속의 추세를 보기 위함."""
    edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        m = idx == i
        if m.sum() >= min_count:
            rows.append({"구간 중앙": (edges[i] + edges[i + 1]) / 2, "평균": float(np.mean(y[m])), "개수": int(m.sum())})
    return pd.DataFrame(rows)


def format_poly(coef, x_name="x", digits=4) -> str:
    terms = []
    for p, c in enumerate(coef):
        if p == 0:
            terms.append(f"{c:.{digits}g}")
        elif p == 1:
            terms.append(f"{c:+.{digits}g}·{x_name}")
        else:
            terms.append(f"{c:+.{digits}g}·{x_name}^{p}")
    return "y = " + " ".join(terms)
