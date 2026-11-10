# 🌊 해양 관측 데이터 회귀분석 · 인공신경망 · 대화형 AI 프로그램

- 데이터처리개론 과제 1 — 선형회귀와 비선형회귀 프로그래밍 실습
- 데이터처리개론 과제 2 — 인공신경망을 이용한 데이터 예측
- 데이터처리개론 과제 3 — 오픈소스 LLM 설치 및 실행

주제: 해상 풍속과 유의파고의 관계 분석 (기상청 해양기상부이, 거문도 1년 시간자료)

- **실행 링크**: https://jsqmam-del.github.io/wind-wave-regression/
- **설명서(Notion)**
  - 과제 1: https://citrine-language-2c7.notion.site/3e80be794f3080059649e540980aa316
  - 과제 2: https://citrine-language-2c7.notion.site/3f30be794f3081da83cfda1998e64f64

## 사용법
1. 실행 링크를 열면 거문도 데이터가 자동으로 불러와집니다. 다른 CSV·Excel 파일은 "내 파일 열기"로 선택합니다.
2. X축, Y축으로 쓸 열을 고릅니다.
3. 선형회귀 / 비선형회귀 / 인공신경망 탭에서 결과를 확인합니다.
4. 대화형 AI 탭에서 모델을 골라 "모델 불러오기"를 누른 뒤 질문합니다. (PC의 Chrome 또는 Edge 필요, 처음 한 번은 모델 파일을 내려받음)

## 기능
- CSV(UTF-8, CP949) / Excel 불러오기, X·Y축 선택, 결측·오류값 제거
- 선형회귀: 경사하강법 직접 구현(학습률·반복 횟수 조절), 최소제곱법 공식과 결과 비교, 손실 감소 그래프, 잔차 그래프
- 비선형회귀: 다항 회귀(1~8차), 거듭제곱 모델(로그 변환 후 최소제곱), 구간별 평균, 차수별 학습/테스트 R²로 과적합 확인
- 인공신경망: 다층 퍼셉트론 직접 구현(순전파·역전파·미니배치 경사하강법·Adam), 여러 입력 변수 선택(풍속·풍향 sin/cos·파주기·과거 풍속·24시간 평균 풍속), 정답과 거의 같은 열 자동 경고, 하루 단위 학습/검증/테스트 분할과 조기 종료, 손실 곡선·가중치 그림·예측 vs 실제·순열 중요도·시계열 비교·실험 기록
- 대화형 AI: 오픈소스 LLM(Qwen2.5 0.5B·1.5B, Llama 3.2 1B, Gemma 2 2B)을 오픈소스 엔진 WebLLM으로 브라우저에서 직접 실행(WebGPU, 서버·API 키 없음), 글자가 실시간으로 나오는 답변과 멈춤, 과제 1·2 분석 결과를 AI에게 알려주는 연동, 응답 시간·초당 토큰 테스트 기록, 대화 내용 저장

## 파일 구조
- `index.html` : 프로그램 전체 (화면 + 회귀 알고리즘 + 인공신경망 + 대화형 AI, JavaScript)
- `geomundo_buoy.csv` : 기상청 기상자료개방포털 해양기상부이 거문도(22103) 시간자료
- `app.py`, `regression.py` : 과제 1 회귀 알고리즘의 Python 버전 (Streamlit)

## 출처와 라이선스
| 구분 | 이름 | 만든 곳 | 라이선스 |
| --- | --- | --- | --- |
| 데이터 | 해양기상부이 거문도(22103) 시간자료 | 기상청 기상자료개방포털 (https://data.kma.go.kr) | 공공데이터 (출처 표시) |
| 언어모델 | Qwen2.5-1.5B-Instruct, Qwen2.5-0.5B-Instruct | Alibaba Cloud (Qwen 팀) | Apache License 2.0 |
| 언어모델 | Llama-3.2-1B-Instruct | Meta | Llama 3.2 Community License — Built with Llama |
| 언어모델 | Gemma-2-2B-it | Google | Gemma Terms of Use |
| 실행 엔진 | WebLLM 0.2.79 (https://github.com/mlc-ai/web-llm) | MLC AI | Apache License 2.0 |
| 모델 파일 | 브라우저용 변환 모델 (https://huggingface.co/mlc-ai) | MLC AI | 원본 모델의 라이선스를 따름 |
| 라이브러리 | Plotly.js 2.35.2 (그래프) | Plotly | MIT License |
| 라이브러리 | SheetJS 0.20.3 (엑셀 읽기) | SheetJS | Apache License 2.0 |

- 모든 구성 요소는 무료이며, 사이트 이용에 로그인·결제·API 키가 필요하지 않습니다.
- 언어모델은 방문자의 브라우저(WebGPU)에서 실행되며 질문 내용은 외부 서버로 전송되지 않습니다.
- 회귀분석과 인공신경망 알고리즘은 외부 라이브러리 없이 직접 구현했습니다.
