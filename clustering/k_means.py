import numpy as np
import matplotlib.pyplot as plt

def kMeans(X, K, iterations):
	(m, n) = X.shape

	idx = np.random.randint(m, size = K)
	mu = X[idx, :]
	#print((np.sum((X[1] - mu)**2, axis = 1)).shape)

	C = np.zeros((m, 1)) #The ith row would hold the cluster index assigned to the ith data point

	for iters in range(iterations):
		for i in range(m):
			C[i,0] = np.argmin(np.sum((X[i] - mu)**2, axis = 1, keepdims=True))

		for k in range(K):
			indices = np.where(C == np.array([k]))[0]
			x_rows = X[indices, :]

			mu[k] = np.mean(x_rows, 0, keepdims = True)



	return C, mu

#----------------------------------------------------------------------

#Data

X = np.array([[1.0,3.0],[9.3,6.8],[1.1,5.3],[2.5,13.2],[1.4,5.6],[2.0,2.0],[2.5,4.39],[9.0,3.0],[3.5,14.5],[1.11,3.78],[4.0,10.0],[1.6,11.58],[9.35,4.5],[4.2,12.4],[2.0,3.0],[2.3,1.0],[3.4,2.2],[1.1,3.4],[1.25,3.4],[1.45,2],[2.3,2.11],[3.2,2.22],[1.22,1.2],[9.5,6.7],[1.2,15.3],[2.55,10.3],[2.6,11],[2.6,11.2],[2.7,11.3],[1.9,12.2]])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


x = X[:, 0]
y = X[:, 1]

#ax.scatter(x, y)
#plt.show()

#-----------------------------------------------------------------------------------------------------------------

C, mu = kMeans(X, 3, 200)

colors = ['b', 'g', 'r']

#print(C)

for k in range(mu.shape[0]):
	indices = np.where(C == [k])[0]
	x_rows = X[indices, :]
	ax.scatter(x_rows[:,0], x_rows[:,1], c=colors[k], label=str(k))


plt.show()
