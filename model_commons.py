import seaborn
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from ml_models import CML
import numpy as np


class ModelCommons(CML):
    def __init__(self):
        super().__init__()
        self.model_name = None
        pass

    def calc_all_metrics(self):
        self.model_accuracy()
        self.conf_matrix()
        self.model_precision()
        self.model_recall()
        self.f1_score()
        self.plot_roc()
        self.pca_on = None

    def calc_all_ann_metrics(self):
        self.ann_model_metrics()
        self.ann_conf_matrix()
        self.plot_train_val_loss()
        self.pca_on = None

    def model_accuracy(self):
        self.model_name = str(type(self.model.best_estimator_).__name__)
        if self.pca_on is None:
            model_accuracy = self.model.score(self.x_test, self.y_test)
            print(self.model_name + "'s accuracy comes out to be: ", model_accuracy)
        else:
            transformed_test = self.pca.transform(self.x_test)
            model_accuracy = self.model.score(transformed_test, self.y_test)
            print(self.model_name + " - Reduced Feature's accuracy comes out to be: ", model_accuracy)

    def conf_matrix(self):
        if self.pca_on is None:
            matrix = confusion_matrix(self.y_test, self.model.predict(self.x_test),
                                      labels=['female', 'male'])
        else:
            transformed_test = self.pca.transform(self.x_test)
            matrix = confusion_matrix(self.y_test, self.model.predict(transformed_test), labels=['female', 'male'])
        seaborn.heatmap(matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['female', 'male'],
                        yticklabels=['female', 'male'])
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(self.model_name + ' Confusion Matrix')
        plt.savefig('./Report Images/' + self.model_name + ' Confusion Matrix.png')
        plt.show()

    def model_precision(self):
        if self.pca_on is None:
            precision = precision_score(self.y_test, self.model.predict(self.x_test), pos_label='female')
        else:
            transformed_test = self.pca.transform(self.x_test)
            precision = precision_score(self.y_test, self.model.predict(transformed_test), pos_label='female')
        print(self.model_name + "'s precision comes out to be: ", precision)

    def model_recall(self):
        if self.pca_on is None:
            recall = recall_score(self.y_test, self.model.predict(self.x_test), pos_label='female')
        else:
            transformed_test = self.pca.transform(self.x_test)
            recall = recall_score(self.y_test, self.model.predict(transformed_test), pos_label='female')
        print(self.model_name + "'s recall comes out to be: ", recall)

    def f1_score(self):
        if self.pca_on is None:
            f1 = f1_score(self.y_test, self.model.predict(self.x_test), pos_label='female')
        else:
            transformed_test = self.pca.transform(self.x_test)
            f1 = f1_score(self.y_test, self.model.predict(transformed_test), pos_label='female')
        print(self.model_name + "'s F1-Score comes out to be: ", f1)

    def ann_model_metrics(self):
        if self.pca_on is None:
            self.model_name = "Feed Forward Neural Network"
            y_pred = self.model.predict(self.x_test)
            y_pred_binary = (y_pred >= 0.5).astype(int)
            loss, model_accuracy = self.model.evaluate(self.x_test, self.y_test)
            print(self.model_name + "'s accuracy comes out to be: ", model_accuracy)
        else:
            self.model_name = "Feed Forward Neural Network (Reduced Dimensions)"
            transformed_test = self.pca.transform(self.x_test)
            y_pred = self.model.predict(transformed_test)
            y_pred_binary = (y_pred >= 0.5).astype(int)
            loss, model_accuracy = self.model.evaluate(transformed_test, self.y_test)
            print(self.model_name + " - Reduced Feature's accuracy comes out to be: ", model_accuracy)

        precision = precision_score(self.y_test, y_pred_binary, pos_label=0)
        print(self.model_name + "'s precision comes out to be: ", precision)

        recall = recall_score(self.y_test, y_pred_binary, pos_label=0)
        print(self.model_name + "'s recall comes out to be: ", recall)

        f1 = f1_score(self.y_test, y_pred_binary, pos_label=0)
        print(self.model_name + "'s F1-Score comes out to be: ", f1)

    def plot_train_val_loss(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.savefig('./Report Images/' + self.model_name + ' Training-Validation Accuracy.png')
        plt.show()

    def ann_conf_matrix(self):
        if self.pca_on is None:
            y_pred = self.model.predict(self.x_test)
            y_pred_binary = (y_pred >= 0.5).astype(int)
            matrix = confusion_matrix(self.y_test, y_pred_binary)
        else:
            transformed_test = self.pca.transform(self.x_test)
            y_pred = self.model.predict(transformed_test)
            y_pred_binary = (y_pred >= 0.5).astype(int)
            matrix = confusion_matrix(self.y_test, y_pred_binary)
        seaborn.heatmap(matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['female', 'male'],
                        yticklabels=['female', 'male'])
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(self.model_name + ' Confusion Matrix')
        plt.savefig('./Report Images/' + self.model_name + ' Confusion Matrix.png')
        plt.show()

    def plot_roc(self):
        if self.pca_on is None:
            predict_proba = self.model.predict_proba(self.x_test)
        else:
            transformed_test = self.pca.transform(self.x_test)
            predict_proba = self.model.predict_proba(transformed_test)

        fpr, tpr, thresholds = roc_curve(self.y_test, predict_proba[:, 0], pos_label='female')
        roc_auc = auc(fpr, tpr)

        # plot the ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title(self.model_name + ' ROC Curve')
        plt.savefig('./Report Images/' + self.model_name + ' ROC Curve.png')
        plt.show()
