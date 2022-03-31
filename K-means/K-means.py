import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Mall.csv')

print(data.info())

cluster_data = data.iloc[:,2:4]
cluster_data = StandardScaler().fit_transform(cluster_data)

K = range(1, 18)
distortion = [KMeans(n_clusters=k).fit(cluster_data).inertia_ for k in K]

plt.plot(K, distortion, 'bx-')
plt.xlabel('k')
plt.ylabel('distortion')
plt.show()

k = 4
kmeans = KMeans(n_clusters=k).fit(cluster_data)
x = data.iloc[:,2:4].values
labels = kmeans.fit_predict(x)

color = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(k):
    plt.scatter(x[labels == i,0], x[labels == i,1], c=color[i], label=('Cluster' + str(i)))
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', marker='X', label='Centroids')
plt.show()
