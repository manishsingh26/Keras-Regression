import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


class DataAnalysis(object):

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_csv(self.file_path)

        df = df.rename(index=str, columns={"GRE Score": "GRE", "TOEFL Score": "TOEFL", "Chance of Admit ": "Admission_Chance"})
        df = df.drop("Serial No.", axis=1)

        print("Data Description :: ")
        print(df.describe())

        X = np.asarray(df.drop("Admission_Chance", axis=1))
        Y = np.asarray(df["Admission_Chance"])
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        return X_train, X_test, y_train, y_test


class NeuralConfiguration(object):

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def model_creation(self):

        print("Creating Sequential Model.")
        model = Sequential()
        model.add(Dense(16, input_dim=7, activation='relu'))
        model.add(Dense(8, input_dim=7, activation='relu'))
        model.add(Dense(1))

        print("Adding Optimizer in the Model.")
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        print("Fitting Data in th Model.")
        model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), nb_epoch=50, batch_size=100)

        print("Working on Model Evaluation.")
        _, accuracy = model.evaluate(self.X_train, self.y_train)
        print('Accuracy: %.2f' % (accuracy * 100))

        print("Predicting Value for the Test Data-set.")
        predicted_data = model.predict(np.array(self.X_test))
        return predicted_data


if __name__ == "__main__":

    file_path_ = r"C:\Users\m4singh\Documents\AnalysisNoteBook\DeepLearning\Regression\Admission_Predict_Ver1.1.csv"

    file_obj = DataAnalysis(file_path_)
    X_train_, X_test_, y_train_, y_test_ = file_obj.load_data()

    neural_obj = NeuralConfiguration(X_train_, X_test_, y_train_, y_test_)
    predicted_data_ = neural_obj.model_creation()

    print(predicted_data_[:4])
