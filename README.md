# the-cleora
Semantic Web and Social Networks Final Project at @PUT Poznan:
KG Embeddings: exploring **Cleora**: A Simple, Strong and Scalable Graph Embedding Scheme.

### Case study
We take on the role of a recruiter aiming to identify top talent for machine learning engineer positions. Our goal is to focus on active and engaged users, specifically those with at least 10 starred repositories, as these individuals will likely be better candidates. Since the dataset we are using, GitHub Social Network [1], was created in June 2019, these developers likely now have even more experience and expertise. <br></br>
For node embeddings, we will use Cleora [2]embeddings, which capture structural information by embedding relationships among nodes.


### About the dataset

A large social network of GitHub developers which was collected from the public API in June 2019. Nodes are developers who have starred at least 10 repositories and edges are mutual follower relationships between them. The vertex features are extracted based on the location, repositories starred, employer and e-mail address. The task related to the graph is binary node classification - one has to predict whether the GitHub user is a web or a machine learning developer. This target feature was derived from the job title of each user.


### Citations
1. Benedek Rozemberczki and Carl Allen and Rik Sarkar, **Multi-scale Attributed Node Embedding**, 2019
2. Barbara Rychalska and Piotr Bąbel and Konrad Gołuchowski and Andrzej Michałowski and Jacek Dąbrowski, **Cleora: A Simple, Strong and Scalable Graph Embedding Scheme**, 2021


## How to run

Run main.py file in python 3.8 or above. If you want to change parameters look at src/config file.
