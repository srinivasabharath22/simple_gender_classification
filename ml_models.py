from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from preprocessing import PreProcessing


class CML(PreProcessing):

    def __init__(self):
        super().__init__()
        self.model = None
        self.history = None
        pass

    def linear_classifier(self):
        C = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        penalty = ['l1', 'l2']

        hyperparams = dict(C=C, penalty=penalty)

        sgdCV = LogisticRegression(random_state=42)
        self.model = GridSearchCV(sgdCV, hyperparams, cv=4, verbose=1)
        self.model.fit(self.x_train, self.y_train)

    def k_nearest_neighbors(self):
        neighbors = list(range(1, 10))
        algos = ['auto', 'ball_tree', 'kd_tree', 'brute']
        distance_params = [1, 2]

        hyperparams = dict(n_neighbors=neighbors, p=distance_params, algorithm=algos)

        knnCV = KNeighborsClassifier()
        self.model = GridSearchCV(knnCV, hyperparams, cv=4, verbose=1)
        self.model.fit(self.x_train, self.y_train)

    def k_nearest_reduced_features(self):
        neighbors = list(range(1, 10))
        algos = ['auto', 'ball_tree', 'kd_tree', 'brute']
        distance_params = [1, 2]

        hyperparams = dict(n_neighbors=neighbors, p=distance_params, algorithm=algos)

        knnCV = KNeighborsClassifier()
        self.model = GridSearchCV(knnCV, hyperparams, cv=4, verbose=1)
        self.model.fit(self.scores, self.y_train)
        self.pca_on = True

    def decision_tree_classifier(self):
        splitter = ['best', 'random']
        criterion = ['gini', 'entropy']
        min_samples_leaf = [1, 2, 4]
        features = ['sqrt', 'log2', None]

        hyperparams = dict(min_samples_leaf=min_samples_leaf, criterion=criterion, max_features=features,
                           splitter=splitter)

        rfcCV = DecisionTreeClassifier(random_state=42)
        self.model = GridSearchCV(rfcCV, hyperparams, cv=4, verbose=1)
        self.model.fit(self.x_train, self.y_train)
        print(self.model.best_params_)
        print('Break')

    def support_vector_classifier(self):
        C = [0.1, 1, 10, 100]
        kernel = ['linear', 'rbf', 'poly', 'sigmoid']
        degree = [1, 2, 3, 4, 5]
        gamma = ['scale', 'auto']

        hyperparams = dict(C=C, kernel=kernel, degree=degree, gamma=gamma)

        svcCV = svm.SVC(probability=True)
        self.model = GridSearchCV(svcCV, hyperparams, cv=4, verbose=1)
        self.model.fit(self.x_train, self.y_train)

    def create_ann_model_full(self, activation='relu', optimizer='adam'):
        model = Sequential()
        model.add(
            Dense(32, input_dim=self.x_train.shape[1], activation=activation)
        )
        model.add(
            Dense(16, activation=activation)
        )
        # model.add(
        #     Dense(64, activation=activation)
        # )
        # model.add(
        #     Dropout(dropout_rate)
        # )
        model.add(
            Dense(1, activation=activation)
        )
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def ann_full(self):
        self.label_encoding()
        ann_model = KerasClassifier(build_fn=self.create_ann_model_full, verbose=1)
        parameters = {'batch_size': [16, 32],
                      'epochs': [50, 100],
                      'optimizer': ['adam', 'rmsprop'],
                      'activation': ['relu', 'sigmoid']
                      }
        self.model = GridSearchCV(estimator=ann_model,
                                  param_grid=parameters,
                                  scoring='accuracy',
                                  cv=4)
        print('START')
        self.model.fit(self.x_train, self.y_train, validation_split=0.1)
        y_pred = self.model.best_estimator_.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print('Break')

    def create_ann_scores(self, activation='relu', optimizer='adam'):
        model = Sequential()
        model.add(
            Dense(64, input_dim=self.scores.shape[1], activation='relu')
        )
        model.add(
            Dense(32, activation='relu')
        )
        model.add(
            Dense(1, activation='sigmoid')
        )
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def ann_scores(self):
        self.label_encoding()
        ann_model = KerasClassifier(build_fn=self.create_ann_scores, verbose=1)
        parameters = {'batch_size': [8, 16, 32],
                      'epochs': [50, 100],
                      'optimizer': ['adam', 'rmsprop'],
                      'activation': ['relu', 'sigmoid']
                      }
        self.model = GridSearchCV(estimator=ann_model,
                                  param_grid=parameters,
                                  scoring='accuracy',
                                  cv=3)
        early_stop = EarlyStopping(monitor='val_loss', patience=10)
        self.model.fit(self.scores, self.y_train, validation_split=0.1, callbacks=[early_stop])
        transformed_test = self.pca.transform(self.x_test)
        y_pred = self.model.best_estimator_.predict(transformed_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print('Break')

    def simplest_ann_score_space(self):
        self.label_encoding()
        self.model = Sequential()
        self.model.add(
            Dense(16, input_dim=self.x_train.shape[1], activation='relu')
        )
        self.model.add(
            Dense(8, activation='relu')
        )
        self.model.add(
            Dense(1, activation='sigmoid')
        )
        keras.utils.plot_model(
            self.model,
            to_file="model.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
        )
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history = self.model.fit(self.x_train, self.y_train, epochs=50, batch_size=8, validation_split=0.1)
        # self.pca_on = True
        print('Break')
