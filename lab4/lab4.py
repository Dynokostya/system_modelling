import pandas as pd


def linear_regression_manual(x, y):
    # Кількість спостережень та ознак
    num_observations = len(y)
    num_features = len(x[0])

    # Додаємо стовпець з одиницями для вільного члена
    augmented_x = [[1] + list(row) for row in x]

    # Метод найменших квадратів
    # Обчислення коефіцієнтів за формулою (X^T * X)^(-1) * X^T * y
    # Тут X^T * X
    XT_X = [[0] * (num_features + 1) for _ in range(num_features + 1)]
    for i in range(num_features + 1):
        for j in range(num_features + 1):
            for n in range(num_observations):
                XT_X[i][j] += augmented_x[n][i] * augmented_x[n][j]

    # Тут X^T * y
    XT_y = [0] * (num_features + 1)
    for i in range(num_features + 1):
        for n in range(num_observations):
            XT_y[i] += augmented_x[n][i] * y[n]

    # Розв'язок системи рівнянь (X^T * X) * beta = X^T * y
    # Об'єднуємо матриці XT_X та XT_y для створення розширеної матриці
    augmented_matrix = [row + [XT_y[i]] for i, row in enumerate(XT_X)]

    # Прямий хід
    for i in range(num_features + 1):
        # Ділимо i-й рядок на діагональний елемент XT_X[i][i]
        divisor = augmented_matrix[i][i]
        for j in range(num_features + 2):
            augmented_matrix[i][j] /= divisor

        # Віднімаємо i-й рядок з усіх нижчих рядків, щоб створити нулі нижче діагональних елементів
        for k in range(i + 1, num_features + 1):
            factor = augmented_matrix[k][i]
            for j in range(num_features + 2):
                augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    # Зворотний хід
    for i in range(num_features, -1, -1):
        for k in range(i - 1, -1, -1):
            factor = augmented_matrix[k][i]
            for j in range(num_features + 2):
                augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    # Результати знаходяться в останньому стовпці розширеної матриці
    betas = [augmented_matrix[i][-1] for i in range(num_features + 1)]

    return betas


def multiple_determination_coefficient_manual(x, y, betas):
    # Додаємо стовпець з одиницями для вільного члена
    augmented_x = [[1] + list(row) for row in x]

    # Обчислюємо прогнозовані значення
    y_pred = [sum(augmented_x[i][j] * betas[j] for j in range(len(betas))) for i in range(len(y))]

    # Середнє значення y
    y_mean = sum(y) / len(y)

    # Сума квадратів різниці між фактичними та середніми значеннями
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)

    # Сума квадратів різниці між прогнозованими та фактичними значеннями
    ss_res = sum((yi - y_pred_i) ** 2 for yi, y_pred_i in zip(y, y_pred))

    # Обчислення коефіцієнта множинної детермінації R^2
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


def pairwise_correlation_coefficients_manual(x):
    num_variables = len(x[0])
    correlation_matrix = [[0 for _ in range(num_variables)] for _ in range(num_variables)]

    for i in range(num_variables):
        for j in range(num_variables):
            # Обчислюємо середні значення для кожної змінної
            mean_i = sum(row[i] for row in x) / len(x)
            mean_j = sum(row[j] for row in x) / len(x)

            # Обчислюємо чисельник коефіцієнта кореляції
            numerator = sum((row[i] - mean_i) * (row[j] - mean_j) for row in x)

            # Обчислюємо знаменник коефіцієнта кореляції
            sum_sq_i = sum((row[i] - mean_i) ** 2 for row in x)
            sum_sq_j = sum((row[j] - mean_j) ** 2 for row in x)
            denominator = (sum_sq_i * sum_sq_j) ** 0.5

            # Обчислюємо коефіцієнт кореляції
            if denominator != 0:
                correlation_matrix[i][j] = numerator / denominator
            else:
                correlation_matrix[i][j] = 0

    return correlation_matrix


def partial_correlation_coefficients_manual(x):
    num_variables = len(x[0])
    partial_correlation_matrix = [[0 for _ in range(num_variables)] for _ in range(num_variables)]

    # Розрахунок коефіцієнтів кореляції для всіх пар змінних
    correlation_matrix = pairwise_correlation_coefficients_manual(x)

    # Обчислення коефіцієнтів часткової кореляції
    for i in range(num_variables):
        for j in range(num_variables):
            if i == j:
                # Коефіцієнт часткової кореляції для змінної з самою собою завжди 1
                partial_correlation_matrix[i][j] = 1
            else:
                # Коефіцієнт кореляції між змінними i та j
                r_ij = correlation_matrix[i][j]

                # Коефіцієнти кореляції між i та іншими змінними, крім j
                r_ik = [correlation_matrix[i][k] for k in range(num_variables) if k != j and k != i]

                # Коефіцієнти кореляції між j та іншими змінними, крім i
                r_jk = [correlation_matrix[j][k] for k in range(num_variables) if k != i and k != j]

                # Коефіцієнти кореляції між іншими змінними, крім i та j
                r_kk = [correlation_matrix[k][l] for k in range(num_variables) for l in range(k + 1, num_variables) if k != i and k != j and l != i and l != j]

                # Обчислення детермінантів для чисельника та знаменника формули часткової кореляції
                numerator = r_ij - sum(r_ik[l] * r_jk[l] for l in range(len(r_ik)))
                denominator = (1 - sum(r_ik[l] ** 2 for l in range(len(r_ik)))) * (1 - sum(r_jk[l] ** 2 for l in range(len(r_jk))))

                # Обчислення коефіцієнта часткової кореляції
                if denominator > 0:
                    partial_correlation_matrix[i][j] = numerator / (denominator ** 0.5)
                else:
                    partial_correlation_matrix[i][j] = 0

    return partial_correlation_matrix


def check_quality(r_squared):
    # Перевірка якості моделі
    if r_squared > 0.8:
        quality = "Excellent"
    elif r_squared > 0.5:
        quality = "Good"
    elif r_squared > 0.3:
        quality = "Satisfied"
    else:
        quality = "Needs to be improved"

    return quality


def main():
    # Завантаження даних
    # https://www.kaggle.com/datasets/rehandl23/fifa-24-player-stats-dataset/
    df = pd.read_csv('player_stats.csv', encoding='ISO-8859-1')
    selected_data = df[['height', 'weight', 'age']].head(20)
    X = selected_data[['height', 'weight']]  # features
    y = selected_data['age']  # target

    # Обчислюємо коефіцієнти регресії для оновлених даних
    beta_coefficients = linear_regression_manual(X.values, y.values)
    r_squared = multiple_determination_coefficient_manual(selected_data.values, y.values, beta_coefficients)
    correlations = pairwise_correlation_coefficients_manual(selected_data.values)
    partial_correlations = partial_correlation_coefficients_manual(selected_data.values)
    print('Coefficients:', beta_coefficients)
    print('R squared:', r_squared)
    print("Correlation coefficients:")
    print('Height Weight Age')
    for row in correlations:
        row_str = ' '.join([f'{round(abs(element), 2):.2f}' for element in row])
        print(row_str)
    print("Partial correlations:")
    for row in partial_correlations:
        row_str = ' '.join([f'{round(abs(element), 2):.2f}' for element in row])
        print(row_str)
    print(check_quality(r_squared))


main()
