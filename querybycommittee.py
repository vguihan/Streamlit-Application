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
from modAL.models import ActiveLearner, Committee
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from modAL.batch import uncertainty_batch_sampling
from functools import partial
from pyod.models.knn import KNN
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title=None, page_icon=None, layout='wide', initial_sidebar_state='auto')
st.set_option('deprecation.showPyplotGlobalUse', False)

#Let's setup the interactive components on the left side bar first
st.sidebar.title("Start by clustering the data.")
cluster_slider = st.sidebar.slider(min_value=1, max_value=10, value=8, label="First, select the number of clusters.")
st.sidebar.write('Now you can check your results on the right. Change this setting and the settings that follow to explore the data.')

st.sidebar.title("Change sample size")
slider_samplesize = st.sidebar.slider('Start with 10, but expect to increase the number.', min_value=10, max_value=40, value=30)
st.sidebar.write('You selected:', slider_samplesize)

st.sidebar.title("Choose the committee size")
slider_committeesize = st.sidebar.slider('2 is a nice even number. ', min_value=2, max_value=4, value=2)
st.sidebar.write('You selected:', slider_samplesize)

st.sidebar.write("As you adjust these variables, the graphs on the right will repopulate.")

#Much of the K-Means section was adapate from the following two routines:
#https://www.askpython.com/python/examples/plot-k-means-clusters-python
#https://discuss.streamlit.io/t/clustering-or-classifying-data-into-groups/636/2

#Let's cache the data -- 100,000 rows is a lot!
import time

@st.cache(ttl=24*60*60)
def load_data1():
    data1 = pd.read_csv("unicauca_dataset_trimmed.csv")
    return data1
data1 = load_data1()
df = data1
	
#Just the numeric data to make PCA happy
@st.cache
def load_data2():
    data2 = df.drop(columns=['Flow_ID','Source_IP','Source_Port','Destination_IP','Destination_Port','Protocol','Timestamp','ProtocolName'])
    return data2
data2 = load_data2()
dfnumerics = data2
pca = PCA(2)
 
#Transform the data
dftransformed = pca.fit_transform(dfnumerics)
dfshaped = dftransformed.shape

#kmeands and friends
kmeans = KMeans(n_clusters=cluster_slider, random_state=0).fit(dftransformed)
labels = kmeans.fit_predict(dfnumerics)

#more transformation
cluster_array = (dftransformed, labels)
dflabels = pd.DataFrame(labels, columns=['cluster'])
dftransformed2 = pd.DataFrame(dftransformed, columns=['x','y'])
label = kmeans.labels_

#finally some graphing
clrs = ["red", "seagreen", "orange", "blue", "yellow", "purple"]
n_labels = len(set(labels))
u_labels = np.unique(label)
dfjoiner = pd.concat([dflabels, dftransformed2, df], axis=1)

fig, ax = plt.subplots()
for i in u_labels:
	fig = px.scatter(dfjoiner, x="x", y="y", color="cluster", hover_data=['Flow_ID','Source_IP','Source_Port','Destination_IP','Destination_Port','Protocol','Timestamp','ProtocolName',dfjoiner.index])
ax.legend()

st.title('Our KMeans scatter plot and clustered dataframe. Here, the cluster number functions as the target.')
st.write('The table provides the flow ID and a number of other features that were dropped from the table initially to facilitate the PCA analysis and KMeans clustering. The goal is to provide a little more context to the analyst.')

#graph the cluster
st.plotly_chart(fig)

RANDOM_STATE_SEED = 1
np.random.seed(RANDOM_STATE_SEED)

#Join the PCA data with the labels for classifaction, the labels setup as the target.
dflabels = pd.DataFrame(labels)
dfclustered = pd.concat([dfnumerics, dflabels], axis=1)
dfcomplete = pd.concat([df, dflabels], axis=1)

st.dataframe(dfcomplete)

#Much of the code for this appplication was adapted from the reference example provided by ModAL
#https://modal-python.readthedocs.io/en/latest/content/examples/query_by_committee.html

#Starting the classification
data_array = dfclustered.values

# reducing dimensionality again with PCA
pca = PCA(n_components=2, random_state=RANDOM_STATE_SEED).fit_transform(dfnumerics)
dftransformed = pca
dfshaped = dftransformed.shape

dflabels = pd.DataFrame(labels)
dfclustered = pd.concat([dfnumerics, dflabels], axis=1)

data_array = dfclustered.values
X_raw = data_array[:10000, :4]
y_raw = data_array[:10000, 5]

X_pool = X_raw
y_pool = y_raw

n_queries = slider_samplesize

# initializing Committee members
n_members = slider_committeesize
learner_list = list()

for member_idx in range(n_members):
    # initial training data
    n_initial = 3
    train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
    X_train = X_pool[train_idx]
    y_train = y_pool[train_idx]

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, train_idx, axis=0)
    y_pool = np.delete(y_pool, train_idx)

    # initializing learner
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=X_train, y_training=y_train
    )
    learner_list.append(learner)

# assembling the committee
committee = Committee(learner_list=learner_list)

#fig = make_subplots(rows=1, cols=4)
#with plt.style.context('seaborn-white'):
##    for learner_idx, learner in enumerate(committee):
#        fig = px.scatter(x=pca[:, 0], y=pca[:, 1], color=learner.predict(X_raw), title="Learner no. %d initial predictions" % (learner_idx+1))
#        st.plotly_chart(fig)
        
# visualizing the initial predictions per learner
st.title('Initially, the committee is at odds about its predictions.')
st.text("Each learner appoaches the points a little differently, event though the algorithm is identical.")
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(n_members*6, 3))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_members, learner_idx + 1)
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(X_raw), cmap='viridis', s=50)
        plt.title('Learner no. %d predictions after %d queries' % (learner_idx + 1, n_queries))
st.pyplot()

unqueried_score = committee.score(X_raw, y_raw)

st.title('And the prediction accuracy is not that high.')
st.write("More clusters mean less accuracy. Try adjusting the number of learners and the clusters to influence accuracy. ")
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(6, 3))
    prediction = committee.predict(X_raw)
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
    plt.title('Committee initial predictions, accuracy = %1.3f' % unqueried_score)
st.pyplot()
    
performance_history = [unqueried_score]

# query by committee
n_queries = slider_samplesize
for idx in range(n_queries):
    query_idx, query_instance = committee.query(X_pool)
    committee.teach(
        X=X_pool[query_idx].reshape(1, -1),
        y=y_pool[query_idx].reshape(1, )
    )
    performance_history.append(committee.score(X_raw, y_raw))
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

# visualizing the final predictions per learner
st.title('But as each learner gets more data, it improve its individual predictions.')
st.write("More data improves each learner individually, improving their overall accuracy.")
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(n_members*6, 3))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_members, learner_idx + 1)
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(X_raw), cmap='viridis', s=50)
        plt.title('Learner no. %d predictions after %d queries' % (learner_idx + 1, n_queries))
st.pyplot()

st.title("And overall, the committee's accuracy starts to improve significantly.")
st.write("Does adding more committee members change the accuracy much? How does that tradeoffs like compute time vs. accuracy with very large datasets where the number of false positive or negatives might be hepful impactful? ")
# visualizing the Committee's predictions
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 4))
    prediction = committee.predict(X_raw)
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
    plt.title('Committee predictions after %d queries, accuracy = %1.3f'
              % (n_queries, committee.score(X_raw, y_raw)))
st.pyplot()


# Plot our performance over time.
fig, ax = plt.subplots(figsize=(6, 3), dpi=130)

ax.plot(performance_history)
ax.scatter(range(len(performance_history)), performance_history, s=13)

ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

ax.set_ylim(bottom=0, top=1)
ax.grid(True)

ax.set_title('Incremental classification accuracy')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification Accuracy')
st.pyplot()


st.title('Next steps.')
st.write('There are several potential next steps for the application, but enabling changing the basic algorithm or comparing multiple alogrithms in an ensemble would be good for exploratory work.')

