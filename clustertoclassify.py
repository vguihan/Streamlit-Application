import streamlit as st
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.transforms as transforms
import pickle
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from modAL.models import ActiveLearner
from modAL.batch import uncertainty_batch_sampling
from functools import partial
from pyod.models.knn import KNN
from sklearn.ensemble import RandomForestClassifier
#from pyod.models.cblof import CBLOF
#from pyod.models.feature_bagging import FeatureBagging
#from pyod.models.hbos import HBOS
#from pyod.models.iforest import IForest
#from pyod.models.lof import LOF

#Let's setup the interactive components on the left side bar first
st.sidebar.title("Start by clustering the data.")
cluster_slider = st.sidebar.slider(min_value=1, max_value=10, value=8, label="First, select the number of clusters.")
st.sidebar.write('Now you can check your results on the right. Change this setting and the settings that follow to explore the data.')

st.sidebar.title("Choose your learner.")
option_algorithm = st.sidebar.selectbox(
     'Which learner would you like to try? All three approaches use batch-based uncertainty sampling.',
     ('KNN', 'RandomForest', 'Some Third Algo'))
st.sidebar.write('You selected:', option_algorithm)

st.sidebar.title("Change batch size")
slider_batchsize = st.sidebar.slider('3 is good if you are unsure where to start.', min_value=2, max_value=4, value=3)
st.sidebar.write('You selected:', slider_batchsize)

st.sidebar.title("Change sample size")
slider_samplesize = st.sidebar.slider('90 is a good plat to start (e.g., 90 samples in batches of 3.', min_value=30, max_value=120, value=90)
st.sidebar.write('You selected:', slider_samplesize)

st.sidebar.write("As you adjust these variables, the graphs on the right will repopulate.")

#Much of the K-Means section was adapate from the following two routines:
#https://www.askpython.com/python/examples/plot-k-means-clusters-python
#https://discuss.streamlit.io/t/clustering-or-classifying-data-into-groups/636/2

#Let's cache the data -- 100,000 rows is a lot!
import time

st.title('Our initial dataframe, with several of the more common features in network traffic data flows.')
@st.cache
def load_data1():
    data1 = pd.read_csv("unicauca_dataset_trimmed.csv")
    return data1
data1 = load_data1()
df = data1
st.dataframe(df)
	
#Just the numeric data to make PCA happy
dfnumerics = df.drop(columns=['Flow.ID','Source.IP','Source.Port','Destination.IP','Destination.Port','Protocol','Timestamp','ProtocolName'])
pca = PCA(2)
 
#Transform the data
dftransformed = pca.fit_transform(dfnumerics)
dfshaped = dftransformed.shape

kmeans = KMeans(n_clusters=cluster_slider, random_state=0).fit(dftransformed)
labels = kmeans.fit_predict(dfnumerics)
label = kmeans.labels_
clrs = ["red", "seagreen", "orange", "blue", "yellow", "purple"]
n_labels = len(set(labels))

u_labels = np.unique(label)
fig, ax = plt.subplots()
for i in u_labels:
    ax.scatter(dftransformed[labels == i , 0] , dftransformed[labels == i , 1] , label = i)
ax.legend()


st.title('Our KMeans scatter plot and clustered dataframe. Here, the cluster number functions as the target.')
st.write('The table provides the flow ID and a number of other features that were dropped from the table initially to facilitate the PCA analysis. The goal is to provide a little more context to the analyst.')


#graph the cluster
st.pyplot(fig)

#Join the PCA data with the labels for classifaction, the labels setup as the target.
dflabels = pd.DataFrame(labels)
dfclustered = pd.concat([dfnumerics, dflabels], axis=1)

#Join the labels with the full record for a more complete view for the analyst. 
dfcomplete = pd.concat([df, dflabels], axis=1)
	
#Starting the classification
data_array = dfclustered.values
X_raw = data_array[:100000, :4]
y_raw = data_array[:100000, 5]


#Much of the code for the active learning section was adapted from the modAL example application:
#https://modal-python.readthedocs.io/en/latest/content/examples/ranked_batch_mode.html
st.dataframe(dfclustered)
RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)
pca = PCA(n_components=2, random_state=RANDOM_STATE_SEED)
transformed_iris = pca.fit_transform(X=X_raw)

# Isolate the data we'll need for plotting.
x_component = transformed_iris[:, 0]
y_component = transformed_iris[:, 1]
    
# Plot our dimensionality-reduced (via PCA) dataset.
st.title('Traffic classes after PCA transformation')
plt.figure(figsize=(8.5, 6), dpi=130)
fig, ax = plt.subplots()
ax.scatter(x=x_component, y=y_component, c=y_raw, cmap='viridis', s=50, alpha=8/10)
st.pyplot(fig)

n_labeled_examples = X_raw.shape[0]
training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=3)

X_train = X_raw[training_indices]
y_train = y_raw[training_indices]

# Isolate the non-training examples we'll be querying.
X_pool = np.delete(X_raw, training_indices, axis=0)
y_pool = np.delete(y_raw, training_indices, axis=0)

from sklearn.neighbors import KNeighborsClassifier

# Specify our core estimator. The number of neighbours here could be interactive.
knn = KNeighborsClassifier(n_neighbors=3)

# Pre-set our batch sampling to retrieve 3 samples at a time.This could be interactively set by the analyst.
#BATCH_SIZE = 3
BATCH_SIZE = slider_batchsize
preset_batch = partial(uncertainty_batch_sampling, n_instances=BATCH_SIZE)

# Specify our active learning model. It would be easily possible to expand on aglorithms here.
if option_algorithm == "KNN": 
	learner = ActiveLearner(estimator=knn, X_training=X_train,y_training=y_train, query_strategy=preset_batch)
else:
	learner = ActiveLearner(estimator=RandomForestClassifier() , X_training=X_train, y_training=y_train, query_strategy=preset_batch)

# Plotting data
predictions = learner.predict(X_raw)
is_correct = (predictions == y_raw)
unqueried_score = learner.score(X_raw, y_raw)

st.title('Initially, the learner is not that smart -- the more clusters it has to figure out, the more poorly it performs.')

# Plot our classification results.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
ax.scatter(x=x_component[is_correct],  y=y_component[is_correct],  c='g', marker='+', label='Correct')
ax.scatter(x=x_component[~is_correct], y=y_component[~is_correct], c='r', marker='x', label='Incorrect')
ax.legend(loc='top left')
ax.set_title("ActiveLearner class predictions (Accuracy: {score:.3f})".format(score=unqueried_score))
st.pyplot(fig)

# Pool-based sampling. The raw samples could be set interactively.
#N_RAW_SAMPLES = 80
N_RAW_SAMPLES = slider_samplesize
N_QUERIES = N_RAW_SAMPLES // BATCH_SIZE

performance_history = [unqueried_score]

for index in range(N_QUERIES):
    query_index, query_instance = learner.query(X_pool)

    # Teach our ActiveLearner model the record it has requested.
    X, y = X_pool[query_index], y_pool[query_index]
    learner.teach(X=X, y=y)

    # Remove the queried instance from the unlabeled pool.
    X_pool = np.delete(X_pool, query_index, axis=0)
    y_pool = np.delete(y_pool, query_index)

    # Calculate and report our model's accuracy.
    model_accuracy = learner.score(X_raw, y_raw)
   # st.write('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

    # Save our model's performance for plotting.
    performance_history.append(model_accuracy)
    
    # Plot our performance over time.
    
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

ax.plot(performance_history)
ax.scatter(range(len(performance_history)), performance_history, s=13)

ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=N_QUERIES + 3, integer=True))
ax.xaxis.grid(True)

ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylim(bottom=0, top=1)
ax.yaxis.grid(True, linestyle='--', alpha=1/2)

ax.set_title('Incremental classification accuracy')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification Accuracy')

st.title('But after only a few queries, its classification gets much better.')

st.pyplot(fig)

st.write('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

predictions = learner.predict(X_raw)
is_correct = (predictions == y_raw)

# Plot our updated classification results once we've trained our learner.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
ax.scatter(x=x_component[is_correct],  y=y_component[is_correct],  c='g', marker='+', label='Correct')
ax.scatter(x=x_component[~is_correct], y=y_component[~is_correct], c='r', marker='x', label='Incorrect')

ax.set_title('Classification accuracy after {n} queries (* {batch_size} samples/query = {total} samples): {final_acc:.3f}'.format(
    n=N_QUERIES,
    batch_size=BATCH_SIZE,
    total=N_QUERIES * BATCH_SIZE,
    final_acc=performance_history[-1]
))
ax.legend(loc='lower right')

st.title('Resulting in more correctly labelled data more quickly than manual classification.')
st.pyplot(fig)

array =  (x_component, y_component, is_correct)
array_t = np.transpose(array)
df2 = pd.DataFrame(array_t, columns=['x','y','correct'])
df3 = pd.concat([df2, df], axis=1)
fig = px.scatter(df3, x="x", y="y", color="correct", hover_data=['Flow.ID','Source.IP','Source.Port','Destination.IP','Destination.Port','Protocol','Timestamp','ProtocolName',df3.index])
fig.update_layout(
    title_text='Traffic classes after classification learning'
)
st.title('Matplot is nice (and easy to see), but Plotly can make the presentation even more interactive, showing the analyst key bits of the data on hover.')
st.write('It is also easy to zoom in and out of various parts of the graph, save the graph to PNG and other nice touches.')
st.plotly_chart(fig)

st.title('Which the analyst can then reference in a combined dataframe with the original traffic record, its plotting reduction, and whether it was correct.')
st.dataframe(df3)


    
