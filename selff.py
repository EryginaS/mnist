from sklearn.datasets import load_digits #пакет с датасетом для обучения mnist 8x8

digits = load_digits() #загрузка пакета
print(digits.data.shape)
import matplotlib.pyplot as plt
#пример такого изображения
plt.gray()
plt.matshow(digits.images[1])
plt.show()
#входные значения
digits.data[0, : ]
#нормализуем от -2 до 2
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
X[0,:]
#выводит значит этот нормализованный массив
print (X)
#разбивает массив на тестовые данные и данные для обучения
"""Функция train_test_split в scikit learn добавляет данные рандомно в различные
 базы данных — то есть, функция не берет первые 60% строк для учебного набора, а то, что осталось, использует как тестовый."""
from sklearn.model_selection import train_test_split
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
#вот эта функция делает так , чтобы правильный ответ превратился в вектор ,а  не был числом
import numpy as np
def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect
#а тут мы эту функцию применяем для того, чтобы и тестовый набор и обучаемый наборы были векторами (нам потом будет удобно их сравнивать)
y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)
y_train[0], y_v_train[0]

#Создаем нейросеть
nn_structure = [64, 30, 10] #это список (архитектура сети) 64 нейрона первого слоя, затем 30 скрытого и 10 выходного
# функция активации (сигмоидальная)
def f(x):
    return 1 / (1 + np.exp(-x))
#нам еще нужен тангенс угла(производная этой функции) для градиента
def f_deriv(x):
    return f(x) * (1 - f(x))
#шаг 1. Сеть нуждается в рандомных весах .
import numpy.random as r
def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b
#Это та же самая операция, что и выше , тольяко для дельта w и b
def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b
#прямой ход
def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):#Если первый слой, то весами является x, в противном случае
    #Это выход из последнего слоя
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l))
    return h, z

#значения выходного слоя
def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y - h_out) * f_deriv(z_out)
def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)
#Процесс обучения
def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Начало метода градиентного спуска  {}'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Итерация {} из {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}

            # градиентный шаг
            h, z = feed_forward(X[i, :], W, b)
            # функция ошибки
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])

                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))

                    tri_b[l] += delta[l+1]
        
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])

        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func
def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y
#главная функция
if __name__ == "__main__":
    # загрузка данных
    digits = load_digits()
    X_scale = StandardScaler()
    X = X_scale.fit_transform(digits.data)
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    # конвертируем данные (ответ) в вектор
    y_v_train = convert_y_to_vect(y_train)
    y_v_test = convert_y_to_vect(y_test)
    # установливаем структуру
    nn_structure = [64, 30, 10]
    # передаем в функцию для теста данные для теста
    W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)
    # функция ошибки
    plt.plot(avg_cost_func)
    plt.ylabel('Average J')
    plt.xlabel('Iteration number')
    plt.show()
    #протестим
    y_pred = predict_y(W, b, X_test, 3)
    from sklearn.metrics import accuracy_score
    print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))

