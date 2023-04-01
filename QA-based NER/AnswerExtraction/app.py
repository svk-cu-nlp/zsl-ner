import streamlit as st
from question_answering import QuestionAnswering

st.title('Document Question Answering System')
st.write("Loading the models...")
qa = QuestionAnswering()
st.write('Models Loaded')


document_text = st.text_area("Document Text", "", height=100)
query = st.text_input("Query")


#if st.button("Get Answers From Document"):
if len(document_text.strip()) > 0 and len(query.strip()) > 0:
    st.write('Fetching answer...')
    answers_lines = qa.fetch_answers(query, document_text).splitlines()
    answer_first = answers_lines[0]
    reference_first = answers_lines[1]
    st.write('Check the answer below...with reference text')
    st.header("ANSWER: "+answer_first)
    st.subheader("REFERENCE: "+reference_first)
    #st.markdown()


