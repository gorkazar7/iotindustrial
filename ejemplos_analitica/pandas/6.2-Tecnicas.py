import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
import numpy as np

# 1. Crear DataFrame con valores faltantes (NaN)
data = pd.DataFrame({
    'Edad': [18, 22, np.nan, 50, 30],
    'Ingreso': [15000, 18000, 45000, 52000, np.nan],
    'Categoría': ['A', 'B', 'A', 'C', 'B']
})

# Resultados de los datos originales
#    Edad  Ingreso Categoría
# 0  18.0   15000        A
# 1  22.0   18000        B
# 2   NaN   45000        A
# 3  50.0   52000        C
# 4  30.0     NaN        B

# 2. Rellenar valores faltantes para usar en las transformaciones
data['Edad_filled'] = data['Edad'].fillna(data['Edad'].mean())  # Edad media = (18+22+50+30)/4 = 30.0
data['Ingreso_filled'] = data['Ingreso'].fillna(data['Ingreso'].median())  # Mediana ingreso = 31500

# Resultado:
#    Edad_filled  Ingreso_filled
# 0         18.0         15000.0
# 1         22.0         18000.0
# 2         30.0         45000.0
# 3         50.0         52000.0
# 4         30.0         31500.0

# 3. Min-Max Scaling de Edad (entre 0 y 1)
scaler_minmax = MinMaxScaler()
data['Edad_scaled'] = scaler_minmax.fit_transform(data[['Edad_filled']])

# Min: 18, Max: 50 → Fórmula: (x - 18) / (50 - 18)
# Resultado:
# 0 → (18 - 18)/32 = 0.00
# 1 → (22 - 18)/32 = 0.125
# 2 → (30 - 18)/32 = 0.375
# 3 → (50 - 18)/32 = 1.000
# 4 → (30 - 18)/32 = 0.375

# 4. Z-score Scaling de Ingreso
scaler_zscore = StandardScaler()
data['Ingreso_zscore'] = scaler_zscore.fit_transform(data[['Ingreso_filled']])

# Promedio: 32300, Desviación estándar aprox: 14838
# Resultado (aproximado):
# 0 → (15000 - 32300)/14838 ≈ -1.166
# 1 → (18000 - 32300)/14838 ≈ -0.964
# 2 → (45000 - 32300)/14838 ≈ 0.856
# 3 → (52000 - 32300)/14838 ≈ 1.327
# 4 → (31500 - 32300)/14838 ≈ -0.053

# 5. Discretización de Edad en 3 bins uniformes (18 a 50 → rango 32/3 ≈ 10.67)
# Bins: [18-28.67), [28.67-39.33), [39.33-50]
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
data['Edad_binned'] = discretizer.fit_transform(data[['Edad_filled']])

# Resultado:
# 0 (Edad 18.0)  → bin 0
# 1 (Edad 22.0)  → bin 0
# 2 (Edad 30.0)  → bin 1
# 3 (Edad 50.0)  → bin 2
# 4 (Edad 30.0)  → bin 1

# 6. One-Hot Encoding para columna 'Categoría'
data_encoded = pd.get_dummies(data, columns=['Categoría'], prefix='Cat', dummy_na=True)

# Resultado esperado:
#  Cat_A | Cat_B | Cat_C | Cat_nan
#   1    |   0   |   0   |    0
#   0    |   1   |   0   |    0
#   1    |   0   |   0   |    0
#   0    |   0   |   1   |    0
#   0    |   1   |   0   |    0

# Mostrar DataFrame final
print("\n=== Datos Transformados ===")
print(data_encoded)
