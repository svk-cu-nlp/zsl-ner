# Zero-shot Learning Based Named Entity Recognition for Software Engineering Domain

This repository contains a Python implementation of a zero-shot learning based named entity recognition (NER) module that uses a question-answering mechanism to perform the task. The module is specifically designed for the software engineering domain.
Background

Named Entity Recognition (NER) is a fundamental task in natural language processing (NLP) that involves identifying and categorizing named entities in a given text into predefined categories such as person, organization, location, etc. NER is a challenging task in the software engineering domain as it requires identifying domain-specific entities such as programming languages, libraries, software components, etc.

Zero-shot learning is a machine learning paradigm that enables a model to recognize and classify objects even when it has not seen them before during training. In the context of NER, zero-shot learning can be used to recognize new entities that are not present in the training data.

## Approach

The zero-shot NER module in this repository uses a question-answering mechanism to perform NER. Given a text and a list of categories, the module generates a set of questions for each category and uses a pre-trained language model to answer those questions. The answers are then filtered to identify the entities that belong to the corresponding category. The module does not require any training data or manual annotation of entities.

# QA-Based NER
The directory contain two modules _QuestionGeneration.ipynb_ and _AnswerExtraction_ folder. The _QuestionGeneration.ipynb_ generates the potential questions froma given set of entity types. For example, entity type set = ['Software', 'Hardware', 'Organization'] the possible questions will be 
[_What Software is mentioned?_
_Which Hardware is mentioned?_
_Which Organization is mentioned?_]
The _AnswerExtraction_ folder contains three python files. 
1. question_answering.py: It is about t5 model based question answering mechanism.
2. app.py: It is GUI platform to give the context and the question and get the answer from the context. It uses the _question_answering.py_ to initialize and run the t5 model for question answering.
3. AnswerExtraction_GPT-3.py: It is the gpt-3 based question asnwering system that precisely analyze the context and answer the question that will be the intended named entity. If the entity is not present in the context, it simply says "Not Applicable".
