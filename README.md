# Zero-shot Learning Based Named Entity Recognition for Software Engineering Domain

This repository contains a Python implementation of a zero-shot learning based named entity recognition (NER) module that uses a question-answering mechanism to perform the task. The module is specifically designed for the software engineering domain.
Background

Named Entity Recognition (NER) is a fundamental task in natural language processing (NLP) that involves identifying and categorizing named entities in a given text into predefined categories such as person, organization, location, etc. NER is a challenging task in the software engineering domain as it requires identifying domain-specific entities such as programming languages, libraries, software components, etc.

Zero-shot learning is a machine learning paradigm that enables a model to recognize and classify objects even when it has not seen them before during training. In the context of NER, zero-shot learning can be used to recognize new entities that are not present in the training data.

## Approach

The zero-shot NER module in this repository uses a question-answering mechanism to perform NER. Given a text and a list of categories, the module generates a set of questions for each category and uses a pre-trained language model to answer those questions. The answers are then filtered to identify the entities that belong to the corresponding category. The module does not require any training data or manual annotation of entities.

# QA-Based NER
The directory contain two modules _QuestionGeneration.py_ and _AnswerExtraction_ folder. 
