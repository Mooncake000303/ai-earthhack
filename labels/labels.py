import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

file_path = 'train_data.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Extracting the problem statements
problem_statements = data['problem'].dropna().values

# Vectorizing the text data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(problem_statements)

# Using KMeans clustering to group similar problem statements
# The number of clusters is set arbitrarily; this may need to be adjusted based on results
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

# Assigning the cluster labels to each problem statement
clusters = kmeans.labels_

# Grouping the problem statements by their assigned cluster and identifying unique topics
clustered_problems = {}
for i in range(n_clusters):
    cluster_indices = np.where(clusters == i)[0]
    clustered_problems[f"Cluster {i+1}"] = [problem_statements[idx] for idx in cluster_indices]

# Displaying a sample from each cluster to identify unique topics
sampled_problems = {cluster: problems[0] for cluster, problems in clustered_problems.items()}
sampled_problems

# assignm labels
# Add the cluster labels to the original DataFrame
data['Cluster_Label'] = clusters

# Display the first few rows of the DataFrame to check the assignment
data.head()

