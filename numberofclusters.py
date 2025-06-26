'''
CALCULATE THE IDEAL NUMBER OF CLUSTERS
ELBOW METHOD
'''
# create a list to store inertia values
inercia = []

# create a range of possible cluster numbers
valores_k = range(1,10)

# apply the elbow method
# used to identify the ideal number of clusters
for k in valores_k:
    # initialize the kmeans model
    kmeans = KMeans(n_clusters=k)

    # fit the model to the data
    kmeans.fit(df_imob_normalizado)

    # add the inertia value to the list
    inercia.append(kmeans.inertia_)

# plot the elbow method result
import matplotlib.pyplot as plt
plt.plot(valores_k, inercia)
plt.xlabel('Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.show()