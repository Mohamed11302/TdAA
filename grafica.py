import matplotlib.pyplot as plt

# Valores que quieres representar en el gráfico de barras
valores = [0.558458496, 0.629064700, 0.643453991, 0.694138675, 0.5407, 0.5659, 0.6902, 0.713016321]
etiquetas = ['kNN', 'NaiveBayes', 'Decission\nTree\nClassifier', 'Decission\nTree\nRegressor', 'Neural Network', 'Boosting\nClassifier', 'Random Forest', 'Stacking']

# Posiciones en el eje x para las barras
posiciones = range(len(valores))

# Crear el gráfico de barras
plt.bar(posiciones, valores, tick_label=etiquetas, color='skyblue')
for i, valor in enumerate(valores):
    plt.text(i, valor, f'{valor:.4f}', ha='center', va='bottom')
valores_y = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
plt.yticks(valores_y)
# Agregar etiquetas a los ejes
plt.xlabel('Categorías')
plt.ylabel('Valores')

# Mostrar el gráfico
plt.show()
