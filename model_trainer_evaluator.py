
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class ModelTrainerEvaluator:
    def __init__(self, model, class_weights):
        self.model = model
        self.class_weights = class_weights

    def train(self, X_train, y_train, X_val, y_val, epochs=400, batch_size=8):
        self.history = self.model.fit(
            X_train, 
            y_train, 
            validation_data=(X_val, y_val), 
            epochs=epochs, 
            batch_size=batch_size, 
            class_weight=self.class_weights,  # Use class weights
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)]
        )

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions

    def evaluate(self, y_true, y_pred):
        icd_cases_on_training = [1 if i >= 0.6 else 0 for i in y_pred]
        accuracy = accuracy_score(y_true, icd_cases_on_training)
        print(f"Accuracy: {accuracy:.3f}")
        report = classification_report(y_true, icd_cases_on_training, target_names=['Class 0', 'Class 1'])
        print(report)
        return icd_cases_on_training

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()
