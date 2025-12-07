import os
import streamlit as st

from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 🔑 Streamlit Cloud의 secrets.toml 에서 UPSTAGE_API_KEY를 가져와서 환경변수로 설정
if "UPSTAGE_API_KEY" in st.secrets:
    os.environ["UPSTAGE_API_KEY"] = st.secrets["UPSTAGE_API_KEY"]


@st.cache_resource
def load_rag_chain():
    """Chroma 벡터DB + Upstage LLM을 이용한 RAG 체인 로드"""

    # 1) 임베딩 + 벡터스토어 로드
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

    # chroma_db 폴더는 GitHub 저장소에 같이 올라와 있어야 함
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 2) 프롬프트: 학교도서관 독서지원 사서 역할
    prompt = ChatPromptTemplate.from_template(
        """
# RAG용 프롬프트 템플릿
prompt = ChatPromptTemplate.from_template(
    """
너는 학교도서관에서 학생들의 독서활동을 도와주는 도우미야.
아래 '참고 문서(context)' 내용을 바탕으로, 학생의 질문에 대해
친절하고 구체적인 답변을 한국어로 작성해줘.

가능하면:
- 도서관 이용 규정, 대출/반납/연장 방법
- 책 고르는 방법, 독후감 작성법, 독서 토론 방법
등을 중심으로 설명해 줘.

만약 문서에 정보가 없으면 모르는 부분은 솔직하게 모른다고 말해.

[참고 문서]
{context}

[질문]
{question}
"""
)

# Upstage LLM (기본 Solar 챗 모델)
llm = ChatUpstage()

# RAG 체인 구성
rag_chain = (
    {
        "context": retriever,          # 질문을 retriever에 넣으면 관련 문서 반환
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


[참고 문서]
{context}

[학생의 질문]
{question}
        """
    )

    llm = ChatUpstage()

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# 실제 RAG 체인 준비
rag_chain = load_rag_chain()


# -------------------------
# Streamlit 챗봇 UI 부분
# -------------------------
st.set_page_config(page_title="학교도서관 독서활동 지원 RAG 챗봇", page_icon="📚")

st.title("📚 학교도서관 독서활동 지원 RAG 챗봇")
st.caption("도서관 소장자료와 독서교육 자료를 기반으로 독서 관련 질문에 답해주는 챗봇입니다.")

with st.sidebar:
    st.subheader("ℹ️ 사용 안내")
    st.markdown(
        """
**예시 질문**

- 독서모임 진행 방법에 대해 알려줘.
- 독후감 작성 팁 있을까?
- 독서토론 기법의 종류에 대해 설명해줘.

답변은 미리 인덱싱해 둔 도서관 자료와 독서교육 자료를 우선적으로 활용해서 생성됩니다.
        """
    )

# 채팅 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 지금까지의 대화 보여주기
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
user_input = st.chat_input("독서방법, 독서활동, 도서관 이용 등에 대해 궁금한 것을 물어보세요.")

if user_input:
    # 사용자 메시지 화면에 추가
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # RAG 호출
    with st.chat_message("assistant"):
        with st.spinner("생각 중입니다..."):
            answer = rag_chain.invoke(user_input)
            st.markdown(answer)

    # 어시스턴트 응답도 히스토리에 저장
    st.session_state["messages"].append({"role": "assistant", "content": answer})
