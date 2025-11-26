import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
from numpy.typing import NDArray

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights: NDArray[np.float64] | None = None
        self.bias: float = 0.0
        self.errors_per_epoch: list[int] = []
        self.converged_epoch: int | None = None
        
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
                
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
                
                errors += int(update != 0.0)
            
            self.errors_per_epoch.append(errors)
            
            if errors == 0:
                self.converged_epoch = epoch + 1
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
    X, y, test_size=0.2, random_state=42, stratify=y  # type: ignore
)

# Normaliza os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treina o Perceptron
perceptron = Perceptron(learning_rate=0.001, n_iterations=5000, random_state=42)
perceptron.fit(X_train_scaled, y_train)

# Avaliação
y_test_pred = perceptron.predict(X_test_scaled)

# Acurácia
accuracy = np.mean(y_test_pred == y_test)
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")

# F1-Score
f1 = f1_score(y_test, y_test_pred)
print(f"F1-Score: {f1:.4f}")

# Matriz de confusão
cm = confusion_matrix(y_test, y_test_pred)
print("\nMatriz de Confusão:")
print(cm)

# Pesos e Bias aprendidos
print("\nPesos finais do modelo:")
print(perceptron.weights)

print(f"\nBias final: {perceptron.bias:.4f}")

# Convergência
if perceptron.converged_epoch is not None:
    print(f"\nO perceptron convergiu em {perceptron.converged_epoch} épocas.")
else:
    print("\nO perceptron NÃO convergiu nas iterações máximas.")

# Gráfico de erro por época
plt.plot(perceptron.errors_per_epoch)
plt.xlabel('Época')
plt.ylabel('Erros')
plt.title('Convergência do Perceptron')
plt.grid(True)
plt.show()