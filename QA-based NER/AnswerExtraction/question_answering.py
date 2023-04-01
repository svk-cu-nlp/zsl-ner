import streamlit as st
import pysbd
from transformers import pipeline
from sentence_transformers import CrossEncoder
from  transformers  import  AutoTokenizer, AutoModelWithLMHead, pipeline

class QuestionAnswering:

  def __init__(self):
    model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelWithLMHead.from_pretrained(model_name)    
    self.sentence_segmenter = pysbd.Segmenter(language='en',clean=False)
    self.passage_retreival_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    self.qa_model = pipeline("question-answering",'a-ware/bart-squadv2') 
    
  def fetch_answers(self, question, document):
      document_paragraphs = document.splitlines()
      query_paragraph_list = [(question, para) for para in document_paragraphs if len(para.strip()) > 0 ]
      
      scores = self.passage_retreival_model.predict(query_paragraph_list)
      top_5_indices = scores.argsort()[-3:]
      top_5_query_paragraph_list = [query_paragraph_list[i] for i in top_5_indices ]
      top_5_query_paragraph_list.reverse()
      
      top_5_query_paragraph_answer_list = ""
      count = 1
      for query, passage in top_5_query_paragraph_list:
       passage_sentences = self.sentence_segmenter.segment(passage)
       answer = self.qa_model(question = query, context = passage)['answer']
       evidence_sentence = ""
       for i in range(len(passage_sentences)):
           if answer.startswith('.') or answer.startswith(':'):
               answer = answer[1:].strip()
           if answer in passage_sentences[i]:
               evidence_sentence = evidence_sentence + " " + passage_sentences[i]
       
                    
       model_input = f"question: {query} context: {evidence_sentence}"
       encoded_input = self.tokenizer([model_input],
                                 return_tensors='pt',
                                 max_length=512,
                                 truncation=True)
                                       
       output = self.model.generate(input_ids = encoded_input.input_ids,
                               attention_mask = encoded_input.attention_mask)
       output_answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
  
       result_str = ""+str(count)+": "+ output_answer +"\n"  
       result_str = result_str + " "+ evidence_sentence + "\n\n"
       top_5_query_paragraph_answer_list += result_str
       count+=1
       
      return top_5_query_paragraph_answer_list

 
