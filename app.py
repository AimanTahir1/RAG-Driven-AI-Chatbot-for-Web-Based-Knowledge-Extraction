import streamlit as st
import rag_langchain_helper


st.title("RAG Question Answering")

url = st.text_input("Enter the URL:")
user_question = st.text_input("Enter your question:")

if 'ragchain' not in st.session_state:
    st.session_state['ragchain'] = None

if st.button("Get Answer"):
    if url and user_question:
        st.session_state['ragchain'] = gen_answer(url, user_question)
        st.write(st.session_state['ragchain'])
    else:
        st.write("Please enter both URL and question.")

if st.session_state['ragchain']:
    answer = st.session_state['ragchain'].invoke(user_question)
    st.write(answer)
