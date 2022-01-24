import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
import plotly.express as px
train_file = pd.read_csv("/home/anna/PycharmProjects/test_problem/train.tsv", sep = '\t')
test_file = pd.read_csv("/home/anna/PycharmProjects/test_problem/test.tsv", sep = '\t')

def main():
    #first we need to prepare the data: chanhe the str format into list of floats, calc the distances, separate into x and y
    def make_list(input_str):
        first_vector = np.array([float(i) for i in input_str.split(",")])
        return first_vector

    def preprocess_data(file):
        file['embedding_x'] = file['embedding_x'].apply(make_list)
        file['embedding_y'] = file['embedding_y'].apply(make_list)

        #using zip is faster in this situation
        file['distance'] = [distance.euclidean(v1, v2) for v1, v2 in
                                  zip(file['embedding_x'], file['embedding_y'])]
        x = file['distance'].values
        y = file['equal'].values

        return x, y


    x_train, y_train = preprocess_data(train_file)
    x_test, y_test = preprocess_data(test_file)

    #by looking at the graph we can approximatly tell the limit of out threshold search
    fig = px.scatter(np.linspace(0, 2, x_train.shape[0]), x_train, color=y_train)
    fig.show()

    #the model is very simple because the data is very simple: it is just threshold classification

    def model(x, threshold):
        if x < threshold:
            return 1
        else:
            return 0

    accuracy_scores = []
    thresholds = np.linspace(1, 1.4, 300)
    for threshold in thresholds:
        preds = np.array([model(x, threshold) for x in x_train])
        accuracy_scores.append(accuracy_score(y_train, preds))

    threshold = thresholds[np.argmax(accuracy_scores)]

    test_preds = np.array([model(x, threshold) for x in x_test])

    return accuracy_score(y_test, test_preds)

if __name__ == '__main__':
    main()
