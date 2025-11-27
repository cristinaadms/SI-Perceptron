import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
        return 1 / (1 + np.exp(-x))
    
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
            
            # Para se não houver mais erros (opcional para sigmoide, mas mantemos a estrutura)
            if errors == 0:
                break
        
        return self
    
    def predict(self, X):
        if self.weights is None:
            raise ValueError("O modelo precisa ser treinado antes de fazer predições")
        linear_output = np.dot(X, self.weights) + self.bias
        activation = self._activation(linear_output)
        return np.where(activation >= 0.5, 1, 0)


# Carrega o dataset
df = pd.read_csv('dataset.csv')

# Separar features e target
X = df.drop('DEATH_EVENT', axis=1).values
y = df['DEATH_EVENT'].values

# Divide em treino e teste
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,
    random_state=42,stratify=y)

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

# Visualização da Estrutura do Perceptron (Estilo Esquemático)
def plot_perceptron_schematic(perceptron, feature_names):
    weights = perceptron.weights
    bias = perceptron.bias
    n_features = len(weights)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1)
    
    # Coordenadas
    input_x = 0.1
    sum_x = 0.5
    act_x = 0.8
    out_x = 1.0
    
    # Espaçamento vertical para os inputs
    y_range = 0.8
    y_start = 0.5 + y_range/2
    y_step = y_range / (n_features - 1) if n_features > 1 else 0
    
    # Desenhar Inputs e Pesos
    for i in range(n_features):
        y = y_start - i * y_step
        
        # Nó de Input
        circle = mpatches.Circle((input_x, y), 0.03, color='#aaccff', ec='k', zorder=10)
        ax.add_patch(circle)
        
        # Nome da feature
        ax.text(input_x - 0.04, y, feature_names[i], ha='right', va='center', fontsize=9)
        
        # Linha do peso (Input -> Soma)
        ax.plot([input_x, sum_x], [y, 0.5], 'k-', alpha=0.3, zorder=1)
        
        # Valor do peso
        w = weights[i]
        mx, my = (input_x + sum_x)/2, (y + 0.5)/2
        if n_features <= 15: 
            ax.text(mx, my, f"{w:.2f}", fontsize=7, ha='center', va='center', 
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

    # Nó de Bias
    bias_y = 0.1
    circle_b = mpatches.Circle((sum_x, bias_y), 0.04, color='#ccffcc', ec='k', zorder=10)
    ax.add_patch(circle_b)
    ax.text(sum_x, bias_y, "Bias\n(b)", ha='center', va='center', fontsize=9)
    
    # Linha do Bias
    ax.plot([sum_x, sum_x], [bias_y + 0.04, 0.5 - 0.06], 'k-', zorder=1)
    ax.text(sum_x + 0.01, (bias_y + 0.5)/2, f"{bias:.2f}", fontsize=9, fontweight='bold')

    # Nó de Soma
    circle_sum = mpatches.Circle((sum_x, 0.5), 0.06, color='#ffcc99', ec='k', zorder=10)
    ax.add_patch(circle_sum)
    # Símbolo de Somatório
    ax.text(sum_x, 0.5, r"$\Sigma$", fontsize=30, ha='center', va='center', zorder=11)
    ax.text(sum_x, 0.6, "Soma\nPonderada", ha='center', va='bottom', fontsize=9, color='#555')

    # Seta Soma -> Ativação
    ax.arrow(sum_x + 0.06, 0.5, (act_x - sum_x - 0.12), 0, head_width=0.0, head_length=0.0, fc='k', ec='k', zorder=1)

    # Nó de Ativação (Sigmoide)
    circle_act = mpatches.Circle((act_x, 0.5), 0.06, color='#ffffcc', ec='k', zorder=10)
    ax.add_patch(circle_act)
    
    # Desenho da Sigmoide (curva S)
    # Cria pontos para o S
    t = np.linspace(-4, 4, 100)
    sig = 1 / (1 + np.exp(-t))
    # Escala para caber no círculo
    sx = act_x + (t / 4) * 0.04  # largura
    sy = 0.5 + (sig - 0.5) * 0.08 # altura
    ax.plot(sx, sy, 'k-', lw=2, zorder=11)
    
    ax.text(act_x, 0.6, "Sigmoide", ha='center', va='bottom', fontsize=9, color='#555')

    # Seta de Saída
    ax.arrow(act_x + 0.06, 0.5, 0.1, 0, head_width=0.02, head_length=0.02, fc='k', ec='k', zorder=1)
    ax.text(out_x, 0.5, "Output", ha='center', va='center', fontsize=10, fontweight='bold')

    plt.title('Estrutura do Perceptron (Visualização)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Obter nomes das features
feature_names = df.drop('DEATH_EVENT', axis=1).columns.tolist()

# Plotar
plot_perceptron_schematic(perceptron, feature_names)