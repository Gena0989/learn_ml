from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

#from sklearn import tree
#from sklearn.neighbors import KNeighborsClassifier


class ScrappyKNN():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		prediction = []

		for row in X_test:
			lable = self.closest(row)
			prediction.append(lable)
		return prediction

	def closest(self, test_row):
		best_index = 0
		best_dist = self._euc(test_row, self.X_train[best_index])
		for i in range(1, len(self.X_train)):
			dist = self._euc(test_row, self.X_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i

		return self.y_train[best_index]

	@staticmethod
	def _euc(a, b):
		"""Euclidean distance between two points.
		a and b are 1-N dimensional arrays
		"""

		return distance.euclidean(a, b)


iris = datasets.load_iris()

X = iris.data
y = iris.target

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.5)

my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)

prediction = my_classifier.predict(X_test)

print accuracy_score(y_test, prediction)
