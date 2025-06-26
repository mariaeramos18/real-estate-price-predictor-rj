import pandas as pd

# configure generic formatting in pandas
pd.options.display.float_format = '{:,.2f}'.format

# dataset path
endereco = r'C:\Users\alexa\Documents\PROGRAMACAO\Curso Analise de dados Big Data\data'

# DataFrame with real estate data
df_imob = pd.read_csv(endereco + r'\Brasile-real-estate-dataset.csv',
                      sep=';', encoding='ISO-8859-1')

# filter Rio de Janeiro state
df_imob_rj = df_imob[df_imob['state'] == 'Rio de Janeiro']

# filter only houses
df_imob_rj = df_imob_rj[df_imob_rj['property_type'] == 'house']

# Filter RJ (commented line)
# print(df_imob_rj['state'].unique())

# removing missing/null values
df_imob_rj = df_imob_rj.dropna()

'''
CLUSTERING
'''
# variables for clustering
x = ['area_m2', 'price_brl']

# normalize the data
# importing here just to highlight for you!
# then move it to the top! Cool?

# class for normalization
from sklearn.preprocessing import StandardScaler

# create the scaler variable
scaler = StandardScaler()

# normalize df_imob_rj data
df_imob_normalizado = scaler.fit_transform(df_imob_rj[[x[0], x[1]]])

# class for clustering
from sklearn.cluster import KMeans

'''
CALCULATE THE IDEAL NUMBER OF CLUSTERS
ELBOW METHOD
'''
'''# create a list to store inertia values
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

plt.show()'''

# initialize the number of clusters defined by the elbow method
kmeans = KMeans(n_clusters=4)  # the number 4 was identified using the elbow method

# train the normalized dataset
# to find the clusters defined above
kmeans.fit(df_imob_normalizado)

# add the identified cluster to the original dataset
df_imob_rj['cluster'] = kmeans.labels_

# calculate the mean of variables by cluster
df_imob_media = df_imob_rj[['cluster', 'price_brl', 'area_m2']] \
    .groupby('cluster').mean('price_brl', 'area_m2') \
    .reset_index().sort_values('price_brl', ascending=False)

# visualize the means by clusters
print(df_imob_media)

'''
CORRELATION AND SIMPLE LINEAR REGRESSION
'''

# dependent variable
y_price = df_imob_rj['price_brl']

# independent variable
x_m2 = df_imob_rj['area_m2']

# correlation
correlacao = df_imob_rj['price_brl'].corr(df_imob_rj['area_m2'])

# import the train/test split class
# then move it to the top
from sklearn.model_selection import train_test_split

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    x_m2,
    y_price,
    test_size=0.3
)

# import linear regression class
# then move it to the top
from sklearn.linear_model import LinearRegression

# initialize the linear regression model
modelo = LinearRegression()

# train the model
modelo.fit(X_train.values.reshape(-1, 1), y_train)

# apply linear regression to the test data
# make predictions based on the test data
y_pred = modelo.predict(X_test.values.reshape(-1, 1))

# calculate the model score
score = modelo.score(X_test.values.reshape(-1, 1), y_test)

'''
VISUALIZATION
'''
# import matplotlib
# move it to the top later
import matplotlib.pyplot as plt

# create the plot panel
plt.subplots(2, 1, figsize=(15, 10))

# title of the panel
plt.suptitle('Real Estate Data Analysis - RJ\n', fontsize=14)

'''
CLUSTERING PLOT
'''
plt.subplot(2, 1, 1)
plt.scatter(df_imob_rj['price_brl'], df_imob_rj['area_m2'],
            c=df_imob_rj['cluster'], cmap='viridis')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clustering')

# add the color bar
cbar = plt.colorbar()
cbar.set_ticks(df_imob_rj['cluster'].unique())

'''
LINEAR REGRESSION
'''
plt.subplot(2, 1, 2)
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Area')
plt.ylabel('Price')
plt.title(f'Real Estate Prices. Correlation: {correlacao:.2f} Score: {score:.2f}')

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()
