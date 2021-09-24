# Práctico 1 - Clustering
Trabajo de la cátedra "Text Mining". [Descripción del trabajo práctico](https://sites.google.com/unc.edu.ar/textmining2021/pr%C3%A1ctico/clustering?authuser=0).

El objetivo es encontrar grupos de palabras que puedan ser usados como clases de equivalencia en un corpus de texto general en castellano.

El corpus seleccionado son notas periodísticas del diario La Voz. Debido a la limitación de memoria para generar el modelo con SpaCy se necesito recortar el texto original.

Se realizaron diferentes experimentos. En cada uno de ellos se explica el pre-procesamiento realizado y los pasos llevados a cabo en las siguientes *notebooks*: 
- [Clustering con diferentes features](src/features_clustering.ipynb)
- [Embedding neuronal](src/embedding_neuronal.ipynb)
- [LDA](src/lda.ipynb)

## TODO

- Probar tripla como features
- Probar diferentes contextos de la palabra objetivo
- Incluir métricas

## Descripción de las herramientas utilizadas

- [SpaCy](https://spacy.io/)
- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [Gensim](https://radimrehurek.com/gensim/)


## Referencias
- [textmining-clustering](https://github.com/facumolina/textmining-clustering)
- [Clustering de palabras](https://github.com/danibosch/word_clustering)
- [Topic Modeling with Gensim (Python)](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)
- [K Means Clustering Example with Word2Vec in Data Mining or Machine Learning](https://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/)
- [Gensim - Creación del modelo de tema LDA ](https://isolution.pro/es/t/gensim/gensim-creating-lda-topic-model/gensim-creacion-del-modelo-de-tema-lda)
- [LDA in Python – How to grid search best topic models?](https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/)
- [Topic modeling visualization – How to present the results of LDA models?](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/)
- [Word2Vec in Gensim Explained for Creating Word Embedding Models (Pretrained and Custom)](https://machinelearningknowledge.ai/word2vec-in-gensim-explained-for-creating-word-embedding-models-pretrained-and-custom/)
- [Word Embedding Tutorial: Word2vec with Gensi](https://www.guru99.com/word-embedding-word2vec.html)
- [Word2Vec and Semantic Similarity using spacy](https://ashutoshtripathi.com/2020/09/04/word2vec-and-semantic-similarity-using-spacy-nlp-spacy-series-part-7/)
- [Document Clustering with Python](http://brandonrose.org/clustering)
- [Natural Language Processing With spaCy in Python](https://realpython.com/natural-language-processing-spacy-python/#part-of-speech-tagging)
