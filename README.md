# RAG 및 멀티 에이전트 시스템 실습 가이드

## 실습 개요

이 실습에서는 Qdrant 벡터 데이터베이스와 IBM Granite 4.0 Micro 모델을 활용하여 3-에이전트 논문 추천 시스템을 구축합니다.

**실습 시간**: 약 1시간
**실습 파일**: `02_rag_exercise.ipynb`

## 시스템 아키텍처

다음 3개의 에이전트가 순차적으로 동작합니다:

1. **검색 에이전트**: 사용자 쿼리를 받아 Qdrant에서 관련 논문 5개를 검색
2. **요약 에이전트**: 검색된 논문들을 분석하고 핵심 내용을 요약
3. **추천 에이전트**: 분석 결과를 바탕으로 가장 유용한 논문 3개를 추천

## 사전 준비 사항

노트북을 열면 다음 항목들이 이미 설정되어 있습니다:

- 필요한 라이브러리 설치 및 임포트
- ArXiv AI 논문 데이터셋 로딩 (500개 샘플)
- Sentence Transformer 임베딩 모델 초기화
- Qdrant 벡터 데이터베이스 구축
- IBM Granite 4.0 Micro 모델 설정

여러분은 핵심 로직만 구현하면 됩니다.

## TODO 1: 3개의 도구 구현

LangChain의 `@tool` 데코레이터를 사용하여 다음 도구들을 구현하세요:

### 1-1. search_papers_tool

```python
@tool
def search_papers_tool(query: str, top_k: int = 5) -> str:
    """Search for relevant papers in Qdrant database"""
    # 구현 필요
```

**구현 내용**:
- `embedding_model.encode(query).tolist()`로 쿼리를 벡터로 변환
- `client.query_points()`로 Qdrant에서 검색 (최신 API 사용)
- `.points`로 결과 리스트 가져오기
- 각 논문의 제목, 연도, 유사도 점수, 요약(150자) 포함
- 보기 좋게 포맷팅된 문자열로 반환

### 1-2. summarize_papers_tool

```python
@tool
def summarize_papers_tool(papers_info: str) -> str:
    """Analyze and summarize the searched papers"""
    # 구현 필요
```

**구현 내용**:
- 입력: 검색된 논문 정보 문자열
- 프롬프트 작성: LLM에게 다음을 분석하도록 요청
  - 핵심 기술과 연구 방법론
  - 실무 적용 가능성
  - 논문 간 연관성과 발전 방향
- `generate_response(prompt, max_tokens=300)`으로 LLM 호출
- 분석 결과 반환

### 1-3. recommend_papers_tool

```python
@tool
def recommend_papers_tool(summary: str, papers_info: str) -> str:
    """Recommend top 3 papers based on analysis"""
    # 구현 필요
```

**구현 내용**:
- 입력: 요약 결과 + 원본 논문 정보
- 프롬프트 작성: 가장 유용한 논문 3개 추천 요청
  - 추천 이유
  - 활용 방법
  - 읽는 순서
- `generate_response(prompt, max_tokens=400)`으로 LLM 호출
- 추천 결과 반환

## TODO 2: 3개의 에이전트 구현

각 에이전트는 상태를 받아 도구를 실행하고 업데이트된 상태를 반환합니다.

### 2-1. search_agent

```python
def search_agent(state: AgentState) -> AgentState:
    """Search agent: Find top 5 relevant papers"""
    # 구현 필요
```

**구현 내용**:
- `search_papers_tool.invoke()`로 논문 검색
- `state["search_results"]`에 검색 결과 저장
- `state["next_step"] = "summarize"` 설정
- 업데이트된 state 반환

### 2-2. summarize_agent

```python
def summarize_agent(state: AgentState) -> AgentState:
    """Summarize agent: Analyze the searched papers"""
    # 구현 필요
```

**구현 내용**:
- `state["search_results"]`에서 검색 결과 가져오기
- `summarize_papers_tool.invoke()`로 분석 수행
- `state["summary"]`에 요약 결과 저장
- `state["next_step"] = "recommend"` 설정
- 업데이트된 state 반환

### 2-3. recommend_agent

```python
def recommend_agent(state: AgentState) -> AgentState:
    """Recommend agent: Suggest top 3 papers"""
    # 구현 필요
```

**구현 내용**:
- `state["summary"]`와 `state["search_results"]` 사용
- `recommend_papers_tool.invoke()`로 추천 수행
- `state["recommendations"]`에 추천 결과 저장
- `state["next_step"] = "end"` 설정
- 업데이트된 state 반환

## TODO 3: LangGraph 워크플로우 구성

StateGraph를 사용하여 에이전트들을 연결합니다.

```python
workflow = StateGraph(AgentState)

# 구현 필요
```

**구현 내용**:

1. **노드 추가**:
   - `workflow.add_node("search", search_agent)`
   - `workflow.add_node("summarize", summarize_agent)`
   - `workflow.add_node("recommend", recommend_agent)`

2. **시작점 설정**:
   - `workflow.set_entry_point("search")`

3. **조건부 엣지 추가**:
   - search 노드에서 router 함수로 다음 단계 결정
   - summarize 노드에서 router 함수로 다음 단계 결정
   - `workflow.add_conditional_edges()` 사용
   - 각 경로에서 다음 노드 또는 END로 이동

4. **최종 엣지**:
   - recommend 노드에서 END로 연결
   - `workflow.add_edge("recommend", END)`

5. **컴파일**:
   - `app = workflow.compile()`

## 테스트 방법

구현을 완료한 후 노트북의 테스트 셀을 실행하세요:

```python
test_query = "deep learning for computer vision in autonomous driving"
result = run_agent_workflow(test_query)
```

**예상 출력**:
- 검색된 5개 논문의 상세 정보
- 논문들의 핵심 분석 내용
- 추천된 3개 논문과 추천 이유

## 주요 개념

### LangChain Tools
- `@tool` 데코레이터로 함수를 도구로 변환
- `.invoke()` 메서드로 도구 실행
- 딕셔너리 형태로 파라미터 전달

### LangGraph StateGraph
- 상태 기반 워크플로우 관리
- 노드: 각 에이전트 함수
- 엣지: 노드 간 연결
- 조건부 엣지: 상태에 따라 동적으로 경로 결정

### Qdrant API
- `client.query_points()`: 벡터 검색 수행
- `.points`: 검색 결과 리스트 반환
- `result.score`: 유사도 점수
- `result.payload`: 논문 메타데이터

## 디버깅 팁

1. **도구 테스트**: 각 도구를 개별적으로 테스트하세요
   ```python
   result = search_papers_tool.invoke({"query": "test query"})
   print(result)
   ```

2. **상태 확인**: 각 에이전트 실행 후 상태를 출력하세요
   ```python
   print(state["search_results"])
   print(state["summary"])
   ```

3. **프롬프트 확인**: LLM 프롬프트가 명확한지 확인하세요

4. **에러 메시지**: Qdrant API 에러는 대부분 `.points` 누락 때문입니다

## 완료 기준

모든 TODO를 구현하고 테스트가 성공적으로 실행되면 완료입니다:

- 5개 논문이 검색됨
- 논문들의 핵심 내용이 분석됨
- 3개 논문이 추천 이유와 함께 제시됨

시간이 남으면 다른 쿼리로도 테스트해보세요:
- "natural language processing for medical diagnosis"
- "reinforcement learning in robotics"
- "graph neural networks for drug discovery"
