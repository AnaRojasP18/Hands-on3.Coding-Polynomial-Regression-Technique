class PolynomialRegression:
    def __init__(self, dataset):
        self.dataset = dataset

    def fit_linear(self):
        n = len(self.dataset)
        sum_x = sum_y = sum_xy = sum_x_squared = 0

        for data in self.dataset:
            x, y = data.values()
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x_squared += x ** 2

        mean_x = sum_x / n
        mean_y = sum_y / n

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
        intercept = mean_y - slope * mean_x

        return intercept, slope

    def fit_quadratic(self):
        n = len(self.dataset)
        sum_x = sum_y = sum_x_squared = sum_x_cubed = sum_x_fourth = sum_xy = sum_x_squared_y = 0

        for data in self.dataset:
            x, y = data.values()
            sum_x += x
            sum_y += y
            sum_x_squared += x ** 2
            sum_x_cubed += x ** 3
            sum_x_fourth += x ** 4
            sum_xy += x * y
            sum_x_squared_y += (x ** 2) * y

        A = [[n, sum_x, sum_x_squared],
             [sum_x, sum_x_squared, sum_x_cubed],
             [sum_x_squared, sum_x_cubed, sum_x_fourth]]
        B = [sum_y, sum_xy, sum_x_squared_y]

        coefficients = self.solve_equations(A, B)
        return coefficients

    def fit_cubic(self):
        n = len(self.dataset)
        sum_x = sum_y = sum_x_squared = sum_x_cubed = sum_x_fourth = sum_x_fifth = sum_x_sixth = sum_xy = sum_x_squared_y = sum_x_cubed_y = 0

        for data in self.dataset:
            x, y = data.values()
            sum_x += x
            sum_y += y
            sum_x_squared += x ** 2
            sum_x_cubed += x ** 3
            sum_x_fourth += x ** 4
            sum_x_fifth += x ** 5
            sum_x_sixth += x ** 6
            sum_xy += x * y
            sum_x_squared_y += (x ** 2) * y
            sum_x_cubed_y += (x ** 3) * y

        A = [[n, sum_x, sum_x_squared, sum_x_cubed],
             [sum_x, sum_x_squared, sum_x_cubed, sum_x_fourth],
             [sum_x_squared, sum_x_cubed, sum_x_fourth, sum_x_fifth],
             [sum_x_cubed, sum_x_fourth, sum_x_fifth, sum_x_sixth]]
        B = [sum_y, sum_xy, sum_x_squared_y, sum_x_cubed_y]

        coefficients = self.solve_equations(A, B)
        return coefficients

    def solve_equations(self, A, B):
        n = len(A)
        for i in range(n):
            pivot = A[i][i]
            for j in range(i + 1, n):
                ratio = A[j][i] / pivot
                B[j] -= ratio * B[i]
                for k in range(i, n):
                    A[j][k] -= ratio * A[i][k]

        coefficients = [0] * n
        for i in range(n - 1, -1, -1):
            coefficients[i] = B[i]
            for j in range(i + 1, n):
                coefficients[i] -= A[i][j] * coefficients[j]
            coefficients[i] /= A[i][i]

        return coefficients

    def predict(self, coefficients, x):
        prediction = sum(coeff * (x ** index) for index, coeff in enumerate(coefficients))
        return prediction

    def correlation_and_determination(self, predictions):
        n = len(self.dataset)
        sum_squared_errors = 0
        mean_y = sum(data['y'] for data in self.dataset) / n

        for i, data in enumerate(self.dataset):
            y = data['y']
            sum_squared_errors += (y - predictions[i]) ** 2

        ss_total = sum((data['y'] - mean_y) ** 2 for data in self.dataset)
        ss_residual = sum_squared_errors
        determination = 1 - (ss_residual / ss_total)

        # Cálculo de correlación
        numerator = sum((data['y'] - mean_y) * (pred_y - mean_y) for data, pred_y in zip(self.dataset, predictions))
        denominator = (sum((data['y'] - mean_y) ** 2 for data in self.dataset) * sum((pred_y - mean_y) ** 2 for pred_y in predictions)) ** 0.5
        correlation = numerator / denominator if denominator != 0 else 0

        return correlation, determination

    def print_equation(self, coefficients, degree):
        equation = "y = "
        for i, coeff in enumerate(coefficients):
            equation += f"{coeff}x^{i}" if i > 1 else f"{coeff}x^{i}" if i == 1 else f"{coeff}"
            if i != degree:
                equation += " + "
        print(equation)

    def run(self):
        linear_coeffs = self.fit_linear()
        quad_coeffs = self.fit_quadratic()
        cubic_coeffs = self.fit_cubic()

        print("Linear Regression:")
        self.print_equation(linear_coeffs, 1)
        print("Quadratic Regression:")
        self.print_equation(quad_coeffs, 2)
        print("Cubic Regression:")
        self.print_equation(cubic_coeffs, 3)

        print("\nPredictions:")
        for x_value in [100, 110, 120]:
            linear_prediction = self.predict(linear_coeffs, x_value)
            quad_prediction = self.predict(quad_coeffs, x_value)
            cubic_prediction = self.predict(cubic_coeffs, x_value)
            print(f"x = {x_value}: Linear Prediction = {linear_prediction}, Quadratic Prediction = {quad_prediction}, Cubic Prediction = {cubic_prediction}")

        linear_predictions = [self.predict(linear_coeffs, data['x']) for data in self.dataset]
        quad_predictions = [self.predict(quad_coeffs, data['x']) for data in self.dataset]
        cubic_predictions = [self.predict(cubic_coeffs, data['x']) for data in self.dataset]

        linear_correlation, linear_determination = self.correlation_and_determination(linear_predictions)
        quad_correlation, quad_determination = self.correlation_and_determination(quad_predictions)
        cubic_correlation, cubic_determination = self.correlation_and_determination(cubic_predictions)

        print("\nCorrelation and Determination Coefficients:")
        print(f"Linear: Correlation = {linear_correlation}, Determination = {linear_determination}")
        print(f"Quadratic: Correlation = {quad_correlation}, Determination = {quad_determination}")
        print(f"Cubic: Correlation = {cubic_correlation}, Determination = {cubic_determination}")


# Dataset
dataset = [
    {'x': 108.0, 'y': 95.0},
    {'x': 115.0, 'y': 96.0},
    {'x': 106.0, 'y': 95.0},
    {'x': 97.0, 'y': 97.0},
    {'x': 95.0, 'y': 93.0},
    {'x': 91.0, 'y': 94.0},
    {'x': 97.0, 'y': 95.0},
    {'x': 83.0, 'y': 93.0},
    {'x': 83.0, 'y': 92.0},
    {'x': 78.0, 'y': 86.0},
    {'x': 54.0, 'y': 73.0},
    {'x': 67.0, 'y': 80.0},
    {'x': 56.0, 'y': 65.0},
    {'x': 53.0, 'y': 69.0},
    {'x': 61.0, 'y': 77.0},
    {'x': 115.0, 'y': 96.0},
    {'x': 81.0, 'y': 87.0},
    {'x': 78.0, 'y': 89.0},
    {'x': 30.0, 'y': 60.0},
    {'x': 45.0, 'y': 63.0},
    {'x': 99.0, 'y': 95.0},
    {'x': 32.0, 'y': 61.0},
    {'x': 25.0, 'y': 55.0},
    {'x': 28.0, 'y': 56.0},
    {'x': 90.0, 'y': 94.0},
    {'x': 89.0, 'y': 93.0},
]

regression = PolynomialRegression(dataset)
regression.run()
