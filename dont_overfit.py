'''Project Members:
1) Brij Malhotra
2) Gokul Ravi Kumar
3) Rutuja Vijaykumar Gadekar
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score,roc_curve, auc,roc_auc_score#,plot_roc_curve
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV , train_test_split , cross_val_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from collections import Counter
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


def c_v(train_data, ytrain_data):
	idxs = list(range(5))
	for i in range(5):
		train_idxs = idxs[0:i]+idxs[i+1:]
		X_train = train_data[0]
		y_train = ytrain_data[0]
		for j in train_idxs[1:]:
			X_train = np.vstack([X_train,train_data[j]])
			y_train = np.vstack([y_train,ytrain_data[j]])
		yield X_train, y_train, train_data[i], ytrain_data[i]
		
def train_fun(clf,train_data,ytrain_data,cols):
	cvv = c_v(train_data, ytrain_data)
	train_scores, val_scores = [], []
	while True:
		try:
			X_train, y_train, X_val, y_val = next(cvv)
			# print(X_train.shape, y_train.shape)
			clf.fit(X_train[:,cols], y_train)
			train_score = clf.score(X_train[:,cols],y_train)
			val_score = clf.score(X_val[:,cols],y_val)
			train_scores.append(train_score)
			val_scores.append(val_score)
		except Exception as e:
			break
	return train_scores, val_scores   

def get_cols(features):
	return [val['index'] for val in features]

class MutualInformation:

	def entropy(self, y):
		y_counts = Counter(y)
		y_counts = {k:v/len(y) for k,v in y_counts.items()}
		e = 0
		for k,v in y_counts.items():
			if v == 0:
				return 0
			e+=v*np.log2(v)
		return -e
		
	def cond_entropy(self, x, y):
		h = np.histogram(x, bins=30)
		x_counts, ranges = h[0], h[1]
		init = ranges[0]
		m_inf = 0
		for i, val in enumerate(ranges[1:]):
			prob_x = x_counts[i]/sum(x_counts)
			y_dist = y[np.logical_and(init<=x, x<val)]
			inner_term = self.entropy(y_dist)
			m_inf+= (prob_x * inner_term)
		return m_inf
	
	def get_sorted_features(self, X_train, y): 
		enp_y = self.entropy(y)
		m_infos = []
		for i in range(X_train.shape[1]):
			minf = enp_y - self.cond_entropy(X_train[:,i], y)
			m_infos.append({'index':i, 'info':minf})
		sorted_features = sorted(m_infos, key=lambda x:x['info'], reverse=True)
		return sorted_features


if __name__ == '__main__':
	train = pd.read_csv('train.csv' , index_col = 'id')
	
	train_X_array = train[list(train.columns[1:])].values
	train_y_array = train.target.values
	
	X_train, y_train = shuffle(train_X_array, train_y_array, random_state=0)
	y_double = np.expand_dims(y_train, axis=1)
	
	train_data = list()
	ytrain_data = list()
	for i in range(0,X_train.shape[0],50):
		train_data.append(X_train[i:i+50])
		ytrain_data.append(y_double[i:i+50])
	
	X_train, X_val, y_train, y_val = train_test_split(train_X_array, train_y_array, test_size=0.33, random_state=42)
	
	lg_clf = LogisticRegression(class_weight='balanced', solver='liblinear', C= 0.1, max_iter=10000)
	svm_linear = SVC(kernel='linear', probability=True)
	gnb = GaussianNB()
	tree_weak = tree.DecisionTreeClassifier(max_depth=1)
	tree_strong = tree.DecisionTreeClassifier(max_depth=5)
	svm_rbf = SVC(kernel='rbf', probability=True)
	clfs = {
		"logistic_regression":lg_clf, 
		"svm_linear":svm_linear, 
		"svm_rbf":svm_rbf,
		"tree_depth_1":tree_weak,
		"tree_depth_5":tree_strong,
		"Gnb":gnb
	}
	
	ensemble_lr = AdaBoostClassifier(base_estimator=lg_clf)
	ensemble_svm_linear = AdaBoostClassifier(base_estimator=svm_linear)
	ensemble_svm_rbf = BaggingClassifier(base_estimator=svm_rbf)
	ensemble_weak_tree = AdaBoostClassifier(base_estimator=tree_weak)
	ensemble_strong_tree = BaggingClassifier(base_estimator=tree_strong)
	ensemble_gnb = BaggingClassifier(base_estimator=gnb)
	ensembles = {
		"logistic_regression":ensemble_lr,
		"svm_linear":ensemble_svm_linear,
		"svm_rbf":ensemble_svm_rbf,
		"tree_depth_1":ensemble_weak_tree,
		"tree_depth_5":ensemble_strong_tree,
		"Gnb":ensemble_gnb
	}   
	
	sorted_features = MutualInformation().get_sorted_features(train_X_array, train_y_array)  
	#Comparison of Random Selection of Features vs Features Selected using Mutual Information for Different Classifiers
	fig = plt.figure(figsize=(15,8))
	i = 0
	features_max_scores = {}
	for name, clf in clfs.items():
		random_scores, mi_scores, no_of_features = [], [], []
		ax = fig.add_subplot(2,3,i+1)
		for feats in range(5,300,2):
			mi_cols = get_cols(sorted_features[:feats])
			random_cols = np.random.choice(range(300), feats)
			for j, cols in enumerate((mi_cols, random_cols)):
				temp_train = X_train[:,cols]
				clf.fit(temp_train, y_train)
				score = clf.score(X_val[:,cols],y_val)
				if j == 0:
					mi_scores.append(score)
				else:
					random_scores.append(score)
			no_of_features.append(feats)

		features_max_scores[name] = no_of_features[np.argmax(mi_scores)]
		ax.plot(no_of_features, mi_scores, color='red', label='Mutual Information')
		ax.plot(no_of_features, random_scores, color='blue', label='random')
		ax.legend()
		ax.text(0.1,0.8, name, transform=ax.transAxes)
		ax.set_xlabel('number of features')
		ax.set_ylabel('accuracy_score')
		i+=1
	print("best number of features based on information gain\n", features_max_scores)
	# plt.show()
	 

	#Comparison of Mutual Information based Feature Selection for Different Classifiers 
	fig = plt.figure(figsize=(5,5))
	i = 0
	features_max_scores = {}
	for name, clf in clfs.items():
		random_scores, mi_scores, no_of_features = [], [], []
		ax = fig.add_subplot()
		for feats in range(5,300,5):
			mi_cols = get_cols(sorted_features[:feats])
			random_cols = np.random.choice(range(300), feats)
			for j, cols in enumerate((mi_cols, random_cols)):
				temp_train = X_train[:,cols]
				clf.fit(temp_train, y_train)
				score = clf.score(X_val[:,cols],y_val)
				if j == 0:
					mi_scores.append(score)
				else:
					random_scores.append(score)
			no_of_features.append(feats)

		features_max_scores[name] = no_of_features[np.argmax(mi_scores)]
		ax.plot(no_of_features, mi_scores, label=name)
		#ax.plot(no_of_features, random_scores, color='blue', label='random')
		ax.legend() 
	# plt.show()  
	
	
	#Using logistic regression with L1 regularization
	lg_clf.penalty = 'l1'
	lg_clf.fit(train_X_array,train_y_array)
	nz = np.nonzero(lg_clf.coef_)[1]
	#weights = lg_clf.coef_[nz]
	for a in sorted_features[:10]:
		for f in nz:
			if a['index'] == f:
				print(f,round(lg_clf.coef_[0,f],3),round(a['info'],3))       
	
	#Using K Fold Cross Validation to fine tune the hyperparameters for different classifiers
	params_to_optimise = {
		"logistic_regression":{"params":"C","range":np.arange(0.0001,1,0.001)},
		"svm_linear":{"params":"C","range":np.arange(0.0001,1,0.001)},
		"svm_rbf":{"params":"C","range":np.arange(0.0001,1,0.001)},
		"tree_depth_1":{"params":"max_depth","range":np.arange(1,6,1)}
	}

	best_parameters = {}
	fig = plt.figure(figsize=(15,8))
	i = 0
	for name, _ in params_to_optimise.items():
		clf = clfs[name]
		ax = fig.add_subplot(2,2,i+1)
		cols = get_cols(sorted_features[:features_max_scores[name]])
		params = params_to_optimise[name]["params"]
		r = params_to_optimise[name]["range"]
		train_mean_score, val_mean_score = [], []
		for param_value in r:
			if params == 'C':
				clf.C = param_value
			elif params == 'max_depth':
				clf.max_depth = param_value
			train_scores, val_scores = train_fun(clf, train_data, ytrain_data, cols)
			train_mean, val_mean = np.mean(train_scores), np.mean(val_scores)
			train_mean_score.append(train_mean); val_mean_score.append(val_mean)
		i+=1
		ax.plot(r, train_mean_score, label='Training Accuracy')
		ax.plot(r, val_mean_score, label='Validation Accuracy')
		ax.set_xlabel(params)
		ax.set_ylabel("Mean accuracy over folds")
		ax.text(0.7, 0.7, name, transform=ax.transAxes)
		ax.legend()
		best_parameters[name] = {params:r[np.argmax(val_mean_score)]}
	# plt.show()  
	print("best parmeters for classifiers\n", best_parameters)
	

	#Accuracy of each classifier after evaluating the best hyperparameter.
	r = list(range(5,50,5))
	n_estimators = {}
	for name, clf in ensembles.items():
		cols = get_cols(sorted_features[:features_max_scores[name]])
		scores = []
		for n in r:
			clf.n_estimators = n
			clf.fit(X_train[:, cols], y_train)
			score = clf.score(X_val[:,cols], y_val)
			scores.append(score)
		n_estimators[name] = r[np.argmax(scores)]      
	fig = plt.figure(figsize=(15,8))
	i = 0
	r = list(range(5,100,2))
	ensemble_best = {}
	for name, clf in ensembles.items():
		if name == 'tree_depth_5':
			continue
		ax = fig.add_subplot(3,3,i+1)
		cols = get_cols(sorted_features[:features_max_scores[name]])
		train_mean_score, val_mean_score = [], []
		for param_value in r:
			clf.n_estimators = param_value
			if name in ["logistic_regression", "svm_linear", "svm_rbf"]:
				clf.base_estimator.C = best_parameters[name]['C']
			elif name == 'tree_depth_1':
				name = 'decision_tree'
				clf.base_estimator.max_depth = 3
			train_scores, val_scores = train_fun(clf, train_data, ytrain_data, cols)
			train_mean, val_mean = np.mean(train_scores), np.mean(val_scores)
			train_mean_score.append(train_mean); val_mean_score.append(val_mean)
		i+=1
		ax.plot(r, train_mean_score, label='Training Accuracy')
		ax.plot(r, val_mean_score, label='Validation Accuracy')
		ensemble_best[name] = r[np.argmax(val_mean_score)]
		ax.set_xlabel(params)
		ax.set_ylabel("Mean accuracy over folds")
		ax.text(0.7, 0.7, name, transform=ax.transAxes)
		ax.legend()
	# plt.show()
	print("best number of estimators for ensembles\n", ensemble_best)

	#Comparison of Single Classifiers and Ensemble of Classifiers
	ensemble_best["tree_depth_1"] = 5
	ensemble_best["tree_depth_5"] = 5 
	means = []
	stds = []
	all_means = {}
	fig = plt.figure(figsize=(15,8))
	i = 0
	for name, s_clf in clfs.items():
		ax = fig.add_subplot(2,3,i+1)
		cols = get_cols(sorted_features[:features_max_scores[name]])
		e_clf = ensembles[name]
		e_clf.n_estimators = ensemble_best[name]
		if name in ["logistic_regression", "svm_linear", "svm_rbf"]:
			e_clf.base_estimator.C = best_parameters[name]['C']
			s_clf.C = best_parameters[name]['C']
		scores = train_fun(s_clf, train_data, ytrain_data, cols)
		s_mean, s_std = np.mean(scores), np.std(scores)
		scores = train_fun(e_clf, train_data, ytrain_data, cols)
		e_mean, e_std = np.mean(scores), np.std(scores)
		stds = [s_std, e_std]
		ax.errorbar(["single", "ensemble"], [s_mean, e_mean], yerr=stds, fmt='o')
		ax.set_xlabel("Mean and variance in accuracy scores for k dataset")
		ax.set_ylabel("Accuracy score")
		all_means[name] = (s_mean, e_mean)
		i+=1
	print("Final accuracy using the best set of parameters for classifiers and ensembles\n", all_means)
	plt.show()