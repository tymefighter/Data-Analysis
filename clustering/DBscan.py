import numpy as np
import matplotlib.pyplot as plt
import queue

def distance(A, B):
	return np.sum((A-B)**2)

def DBscan(X, eps, minpts):
	(m, n) = X.shape
	cluster = np.zeros((m), dtype=np.int64) - 1
	c_no = 0
	
	q = queue.Queue()

	for i in range(m):
		if cluster[i] != -1:
			continue
		q.put(i)
		flag = 0
		while q.empty() == False:			
			pts = -1
			idx = []
			a = q.get()
			for j in range(m):
				if distance(X[a], X[j]) <= eps and cluster[j] == -1:
					pts += 1
					idx.append(j)
			if pts < minpts:
				continue
			flag = 1
			for k in idx:
				cluster[k] = c_no
				q.put(k)
		if flag == 1:
			c_no += 1

	return cluster

	
#----------------------------------------------------------------------

#Data

X = np.array([[1.0,3.0],[9.3,6.8],[1.1,5.3],[2.5,13.2],[1.4,5.6],[2.0,2.0],[2.5,4.39],[9.0,3.0],[3.5,14.5],[1.11,3.78],[4.0,10.0],[1.6,11.58],[9.35,4.5],[4.2,12.4],[2.0,3.0],[2.3,1.0],[3.4,2.2],[1.1,3.4],[1.25,3.4],[1.45,2],[2.3,2.11],[3.2,2.22],[1.22,1.2],[9.5,6.7],[1.2,15.3],[2.55,10.3],[2.6,11],[2.6,11.2],[2.7,11.3],[1.9,12.2]])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


x = X[:, 0]
y = X[:, 1]


#-----------------------------------------------------------------------------------------------------------------

cluster = DBscan(X, 5.0, 3)

print(cluster)

colors = ['b', 'g', 'r', 'y']

#print(C)

for k in range(-1, max(cluster)+1):
	indices = np.where(cluster == [k])[0]
	x_rows = X[indices, :]
	ax.scatter(x_rows[:,0], x_rows[:,1], c=colors[k], label=str(k))


plt.show()
