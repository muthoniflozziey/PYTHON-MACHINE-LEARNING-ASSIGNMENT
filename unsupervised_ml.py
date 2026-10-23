from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# customer data
customers = [ [1000,2], [1500,3], [2000,2], [5000,6], [6000,7], [5500,5], [12000,12], [15000,14], [13000,11], [3000,4] ]
# scale the data
scaler = StandardScaler()
scaled_customers = scaler.fit_transform(customers)
# create the KMeans model
KMeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# train the model
clusters = KMeans.fit_predict(scaled_customers)
#  display each customer's cluster
print("customer segmentation results:")
for i, cluster in enumerate(clusters):
    print(f"customer {i + 1} belongs to cluster {cluster}")
#count customer in each cluster
print ("\nTotal customers in each cluster:")
for cluster in range(3):
    total = sum(clusters == cluster)
    print(f"cluster {cluster}: {total} customers")
# display cluster assigned to each customer
for i, cluster in enumerate (KMeans.labels_):
    print(f"customer {i + 1} belongs to cluster {cluster}")
# display the total number of customer in each cluster
print("\nTotal customers in each cluster:")
for cluster in range(3):
    count = sum(KMeans.labels_ == cluster)
    print (f"cluster {cluster}: {count} customers")
import matplotlib.pyplot as plt
# create a scatter plot
plt.figure(figsize=(8, 5))
plt.scatter([customer[0] for customer in customers], [customer[1] for customer in customers], c=KMeans.labels_, cmap="viridis", s=100 )
plt.title("customer segmentation using K-Means")
plt.xlabel("Monthly Spending (KES)")
plt.ylabel("Monthly visits")
plt.colorbar(label="cluster")
plt.show()

