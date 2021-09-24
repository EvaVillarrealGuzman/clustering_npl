import sklearn.manifold
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def read_file(name_file, encoding):
    text_file =  open(name_file, "r", encoding=encoding)
    content = text_file.read()
    text_file.close()
    return content

#Separación del corpus en oraciones y en palabras.
def get_words_and_lemma(doc, min_words_count_in_sent, pos_tag):
    sents = [sent for sent in doc.sents if len(sent) > min_words_count_in_sent] #Se eliminan oraciones con menos de 10 palabras

    words = []
    words_lemma = []
    for sent in sents:
        for word in sent:
            if word.is_alpha and not word.is_stop and word.pos_ in pos_tag:
                words.append(word)
                words_lemma.append(word.lemma_)
    return words, words_lemma


def create_counter_word_vector(words_lemma):
    return Counter(words_lemma) #Conteo de ocurrencias totales de cada palabra

def separate_words_and_features(dictionary):
    features = []
    key_words = {}
    wid = 0
    for d in dictionary:
        if len(d) > 0:
            key_words[d] = wid
            wid += 1
            features.append(dictionary[d])
    return features, key_words

def vectorize_and_normalized_words(features, key_words, is_print):
    v = DictVectorizer(sparse=False)
    x = v.fit_transform(features)
    if is_print:
        print("Vectorización de palabras")
        keysVocab = list(key_words)

        ln = "".ljust(15) + " "
        for a in v.feature_names_:
            ln += a.ljust(15) + " "
        print(ln)
        for idx, a in enumerate(x):
            ln = keysVocab[idx].ljust(15) + " "
            for i in a:
                ln += str(i).ljust(15) + " "
            print(ln)

    norm = x / x.max(axis=0)
    if is_print:
        print()
        print()
        print("Normalización de matriz")
        keysVocab = list(key_words)

        ln = "".ljust(15) + " "
        for a in v.feature_names_:
            ln += a.ljust(15) + " "
        print(ln)
        for idx, a in enumerate(norm):
            ln = keysVocab[idx].ljust(15) + " "
            for i in a:
                ln += str(i).ljust(15) + " "
            print(ln)

    return norm

def delete_columns(matrix_normed):
    variances = np.square(matrix_normed).mean(axis=0) - np.square(matrix_normed.mean(axis=0))
    threshold_v = 0.001
    return np.delete(matrix_normed, np.where(variances < threshold_v), axis=1)

def plot_points(key_words, red_matrix):
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
    matrix_dicc2d = tsne.fit_transform(red_matrix)

    pointsspacy = pd.DataFrame(
        [
            (word, coords[0], coords[1])
            for word, coords in [
                (word, matrix_dicc2d[key_words[word]])
                for word in key_words
            ]
        ],
        columns=["word", "x", "y"]
    )
    pointsspacy.plot.scatter("x", "y", s=10, figsize=(20, 12))
    return pointsspacy, matrix_dicc2d

def plot_region(pointsspacy, x_bounds, y_bounds):
    slice = pointsspacy[
        (x_bounds[0] <= pointsspacy.x) &
        (pointsspacy.x <= x_bounds[1]) &
        (y_bounds[0] <= pointsspacy.y) &
        (pointsspacy.y <= y_bounds[1])
    ]

    ax = slice.plot.scatter(x='x', y='y', s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

def clustering(k, red_matrix):
    km_model = KMeans(n_clusters=k)
    km_model.fit(red_matrix)
    print("Clustering finished")
    return km_model

def graph_results(km_model,red_matrix):
    y_pred = km_model.predict(red_matrix)
    plt.scatter(red_matrix[:, 0], red_matrix[:, 1], c=y_pred, s=50, cmap='plasma')
    plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
    plt.figure(figsize=(15,7.5))
    sns.scatterplot(x=red_matrix[y_pred == 0, 0], y=red_matrix[y_pred == 0, 1],s=50)
    sns.scatterplot(x=red_matrix[y_pred == 1, 0], y=red_matrix[y_pred == 1, 1],s=50)
    sns.scatterplot(x=red_matrix[y_pred == 2, 0], y=red_matrix[y_pred == 2, 1],s=50)
    sns.scatterplot(x=red_matrix[y_pred == 3, 0], y=red_matrix[y_pred == 3, 1],s=50)
    sns.scatterplot(x=red_matrix[y_pred == 4, 0], y=red_matrix[y_pred == 4, 1],s=50)
    sns.scatterplot(km_model.cluster_centers_[:, 0], km_model.cluster_centers_[:, 1],s=500,color='yellow')
    plt.title('Clusters')
    plt.legend()
    plt.show()

def show_results(features,model, key_words):
	# Show results
	c = Counter(sorted(model.labels_))
	print("\nTotal clusters:",len(c))
	for cluster in c:
		print ("Cluster#",cluster," - Total words:",c[cluster])

	# Show top terms and words per cluster
	print("Top terms and words per cluster:")
	print()
	#sort cluster centers by proximity to centroid
	order_centroids = model.cluster_centers_.argsort()[:, ::-1]

	keysFeatures = list(features)
	keysVocab = list(key_words)
	for n in range(len(c)):
		print("Cluster %d" % n)
		print("Frequent terms:", end='')
		for ind in order_centroids[n, :10]:
			print(' %s' % keysFeatures[ind], end=',')

		print()
		print("Words:", end='')
		word_indexs = [i for i,x in enumerate(list(model.labels_)) if x == n]
		for i in word_indexs:
			print(' %s' % keysVocab[i], end=',')
		print()
		print()

	print()

#Assigning word types to tokens, like verb or noun - Part-of-speech (POS) Tagging
def get_features_by_POS_tag(word, features):
    pos = "POS__" + word.pos_
    if not pos in features:
        features[pos] = 0
    features[pos] += 1
    return features

#Assigning syntactic dependency labels, describing the relations between individual tokens, like subject or object. - Dependency Parsing
def get_features_by_DEP_tag(word, features):
    dep = "DEP__" + word.dep_
    if not dep in features:
        features[dep] = 0
    features[dep] += 1
    return features

def get_features_by_context_both_size(doc, word, features, threshold_c):
    contexts = []
    for sent in doc.sents:
        for iw, word_doc in enumerate(sent):
            if word_doc == word: #ver lemmma TODO
                start = max(0, iw - threshold_c)
                end = min(len(sent), iw + threshold_c + 1)
                for pos2 in range(start, end):
                    #if pos2 == iw or not str.isdigit(sent[pos2].lemma_) or  counts[sent[pos2].lemma_]  < min_frequency: #frecuencia
                    if pos2 == iw or sent[pos2].pos_ in ['PUNCT']:
                        continue
                    contexts.append(sent[pos2].text)
    context = '__'.join(contexts)
    if not context in features:
        features[context] = 0
    features[context] += 1
    return features


def get_features_by_context_with_frecuency(doc, word, features, threshold_c, counts, min_frequency):
    contexts = []
    for sent in doc.sents:
        for iw, word_doc in enumerate(sent):
            if word_doc == word: #ver lemmma TODO
                start = max(0, iw - threshold_c)
                end = min(len(sent), iw + threshold_c + 1)
                for pos2 in range(start, end):
                    if pos2 == iw or not str.isdigit(sent[pos2].lemma_) or  counts[sent[pos2].lemma_]  < min_frequency or not sent[pos2].is_alpha or sent[pos2].is_stop: #frecuencia
                        continue
                    contexts.append(sent[pos2].text)
    context = '__'.join(contexts)
    if not context in features:
        features[context] = 0
    features[context] += 1
    return features

def get_features_by_tags(word, features):
    dep = "TAG__" + word.tag_
    if not dep in features:
        features[dep] = 0
    features[dep] += 1
    return features

def print_words_more_frequenty(key_words):
    print("Cantidad total de palabras#", len(key_words))

    # Draw a bar chart
    lst = key_words.most_common(20)
    df = pd.DataFrame(lst, columns=['Palabras', 'Cantidad'])
    df.plot.bar(x='Palabras', y='Cantidad')

def experiment(doc, words ,counts, min_frequency, windows_size,
               cluster_number, is_feature_POS,
               is_feature_DEP, is_feature_tags, is_feature_contexto, is_tripla):
    # Crear diccionario
    dicc = {}

    for word in words:
        lemma = word.lemma_
        # Se filtran palabras con frecuencia menos a la definida en la constante MIN_FREQUENCY
        if counts[lemma] < min_frequency:
            continue
        if not lemma in dicc:
            features = {}
        else:
            features = dicc[lemma]

        # agregamos a las features su POS tag
        if is_feature_POS:
            features = get_features_by_POS_tag(word, features)

        # agregamos a las features su DEP tag
        if is_feature_DEP:
            features = get_features_by_DEP_tag(word, features)

        # la morfología del tag, siendo ésta parseada previamente ya que se encuentran unidas en un solo string
        if is_feature_tags:
            features = get_features_by_tags(word, features)

        # agregamos los contextos, sin orden (con una ventana determinada)
        #if is_feature_tags:
        #    features = get_features_by_context_both_size(doc, word, features, windows_size)

        if is_feature_contexto:
            features = get_features_by_context_with_frecuency(doc, word, features, windows_size, counts, min_frequency)

        # agregamos la tripla de dependencia: palabra__lemma__funcionalidad__palabra-head-del-arbol-de-dependencia-lematizada
        if is_tripla:
            tripla = "TRIPLA__" + lemma + "__" + word.lemma_ + "__" + word.dep_ + "__" + word.head.lemma_
            if not tripla in features:
               features[tripla] = 0
            features[tripla] += 1

        dicc[lemma] = features

    #Separamos las palabras y sus features
    features, key_words = separate_words_and_features(dicc)

    #se muestran las palabras
    #print("Cantidad total de palabras#", len(key_words))
    #print("Cantidad total de palabras#", len(counts))

    #lst = counts.most_common(40)
    #df = pd.DataFrame(lst, columns=['Palabras', 'Cantidad'])
    #df.plot.bar(x='Palabras', y='Cantidad')

    #Vectorizamos las palabras con Sklearn y Normalizamos la matriz
    matrix_normed = vectorize_and_normalized_words(features, key_words, False)

    #Reducimos la dimensionalidad quitando aquellas columnas que tengan poca varianza, ya que no nos aporta demasiada información.
    red_matrix = delete_columns(matrix_normed)

    clusters = clustering(cluster_number, red_matrix)

    #Se muestran los resultados. La lista de palabras que pertenecen a cada cluster se encuentran ordenamos por la cercania con el centroide
    show_results(features, clusters, key_words)
    graph_results(clusters, red_matrix)
