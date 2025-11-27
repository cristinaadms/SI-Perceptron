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
    
    # Ajuste de tamanho da figura para manter proporção
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    ax.set_aspect('equal') # Garante que círculos sejam círculos
    ax.set_xlim(-0.1, 1.3)
    ax.set_ylim(0, 1)
    
    # Paleta de Cores Pastéis (hex)
    colors = {
        'input': '#AEC6CF',    # Pastel Blue
        'bias':  '#77DD77',    # Pastel Green
        'sum':   '#FFB347',    # Pastel Orange
        'act':   '#FDFD96',    # Pastel Yellow
        'border': '#555555',   # Dark Gray for borders
        'text':  '#333333'     # Dark Gray for text
    }
    
    # Coordenadas
    input_x = 0.0
    sum_x = 0.5
    act_x = 0.9
    out_x = 1.2
    
    # Espaçamento vertical para os inputs
    y_range = 0.8
    y_start = 0.5 + y_range/2
    y_step = y_range / (n_features - 1) if n_features > 1 else 0
    
    # Função auxiliar para criar círculos estilizados
    def add_node(x, y, color, radius=0.05, label=None, label_y_offset=0):
        circle = mpatches.Circle((x, y), radius, facecolor=color, 
                               edgecolor=colors['border'], linewidth=2, zorder=10)
        ax.add_patch(circle)
        if label:
            ax.text(x, y + label_y_offset, label, ha='center', va='center', 
                   fontsize=10, color=colors['text'], fontweight='bold')
        return circle

    # Desenhar Inputs e Pesos
    for i in range(n_features):
        y = y_start - i * y_step
        
        # Nó de Input
        add_node(input_x, y, colors['input'], radius=0.035)
        
        # Nome da feature (à esquerda)
        ax.text(input_x - 0.06, y, feature_names[i], ha='right', va='center', 
               fontsize=11, color=colors['text'], fontfamily='sans-serif')
        
        # Linha do peso (Input -> Soma)
        ax.plot([input_x, sum_x], [y, 0.5], color=colors['border'], alpha=0.4, zorder=1)
        
        # Valor do peso
        w = weights[i]
        mx, my = (input_x + sum_x)/2, (y + 0.5)/2
        if n_features <= 15: 
            ax.text(mx, my, f"{w:.2f}", fontsize=8, ha='center', va='center', 
                    color=colors['text'],
                    bbox=dict(facecolor='#f9f9f9', alpha=0.8, edgecolor='none', pad=1))

    # Nó de Bias (ACIMA da soma)
    bias_y = 0.85
    add_node(sum_x, bias_y, colors['bias'], radius=0.045)
    ax.text(sum_x, bias_y, "b", ha='center', va='center', fontsize=14, 
           fontstyle='italic', color=colors['text'])
    ax.text(sum_x, bias_y + 0.07, f"{bias:.2f}", ha='center', va='center', fontsize=10, color=colors['text'])
    
    # Linha do Bias
    ax.plot([sum_x, sum_x], [bias_y - 0.045, 0.5 + 0.06], color=colors['border'], linestyle='--', zorder=1)

    # Nó de Soma
    add_node(sum_x, 0.5, colors['sum'], radius=0.06)
    # Símbolo de Somatório
    ax.text(sum_x, 0.5, r"$\Sigma$", fontsize=28, ha='center', va='center', 
           color=colors['text'], zorder=11)
    ax.text(sum_x, 0.38, "Soma\nPonderada", ha='center', va='top', fontsize=10, color=colors['text'])

    # Seta Soma -> Ativação
    ax.annotate("", xy=(act_x - 0.06, 0.5), xytext=(sum_x + 0.06, 0.5),
                arrowprops=dict(arrowstyle="->", color=colors['border'], lw=2))

    # Nó de Ativação (Sigmoide)
    add_node(act_x, 0.5, colors['act'], radius=0.06)
    
    # Desenho da Sigmoide estilizada
    t = np.linspace(-4, 4, 100)
    sig = 1 / (1 + np.exp(-t))
    sx = act_x + (t / 4) * 0.04
    sy = 0.5 + (sig - 0.5) * 0.08
    ax.plot(sx, sy, color=colors['text'], lw=2.5, zorder=11)
    
    # Texto "Sigmoide" ACIMA do nó
    ax.text(act_x, 0.62, "Sigmoide", ha='center', va='bottom', 
           fontsize=10, color=colors['text'], fontweight='bold')

    # Seta de Saída
    ax.annotate("", xy=(out_x, 0.5), xytext=(act_x + 0.06, 0.5),
                arrowprops=dict(arrowstyle="->", color=colors['border'], lw=2))
    
    ax.text(out_x + 0.02, 0.5, "Output", ha='left', va='center', 
           fontsize=12, fontweight='bold', color=colors['text'])

    plt.title('Estrutura do Perceptron (Visualização)', fontsize=16, fontweight='bold', color=colors['text'], pad=20)
    plt.tight_layout()
    plt.show()

# Obter nomes das features
feature_names = df.drop('DEATH_EVENT', axis=1).columns.tolist()

# Plotar
plot_perceptron_schematic(perceptron, feature_names)