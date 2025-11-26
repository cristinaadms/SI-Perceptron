import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.typing import NDArray

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights: NDArray[np.float64] | None = None
        self.bias: float = 0.0
        self.errors_history: list[int] = []
        
    def _activation(self, x):
        return np.where(x >= 0, 1, 0)
    
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Inicializa pesos e bias
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        # Treinamento
        for epoch in range(self.n_iterations):
            errors = 0
            
            for idx, x_i in enumerate(X):
                assert self.weights is not None
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation(linear_output)
                
                # Atualiza pesos se houver erro
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
                
                errors += int(update != 0.0)
            
            self.errors_history.append(errors)
            
            # Para se não houver mais erros
            if errors == 0:
                break
        
        return self
    
    def predict(self, X):
        if self.weights is None:
            raise ValueError("O modelo precisa ser treinado antes de fazer predições")
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)


# Carrega o dataset
df = pd.read_csv('dataset.csv')

# Separar features e target
X = df.drop('DEATH_EVENT', axis=1).values
y = df['DEATH_EVENT'].values

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Normaliza os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treina o Perceptron
perceptron = Perceptron(learning_rate=0.01, n_iterations=1000, random_state=42)
perceptron.fit(X_train_scaled, y_train)

# Calcula acurácia
y_test_pred = perceptron.predict(X_test_scaled)
accuracy = np.mean(y_test_pred == y_test)
print(f"Acurácia no conjunto de teste: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Visualização da convergência
plt.figure(figsize=(10, 6))
plt.plot(perceptron.errors_history, linewidth=2)
plt.xlabel('Época', fontsize=12)
plt.ylabel('Número de Erros', fontsize=12)
plt.title('Convergência do Perceptron', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()