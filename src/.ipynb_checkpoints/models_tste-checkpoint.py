"""
=============================================================================
UNIVERSIDADE PRESBITERIANA MACKENZIE
Faculdade de Computação e Informática
Disciplina: Inteligência Artificial – 7ºJ SI - Noite
Prof. Dr. Leandro Zerbinatti

Projeto: PREDIÇÃO DE PREÇOS DE ENERGIA COM MACHINE LEARNING
Integrantes: [Nome Completo] - RA: [número] - [email]

Arquivo: analise_exploratoria_modelagem.ipynb
Descrição: Análise exploratória completa, preparação de dados e modelagem
           para predição de preços de energia elétrica

Histórico de Alterações:
- 2025-11-15 | [Nome] | Criação inicial do notebook com análise exploratória
- 2025-11-15 | [Nome] | Implementação de preparação de dados
- 2025-11-15 | [Nome] | Implementação dos modelos de ML
=============================================================================
"""

# =============================================================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# Configurações de visualização
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# Bibliotecas de Machine Learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Dataset
import kagglehub

print("Bibliotecas importadas com sucesso!")

# =============================================================================
# 2. CARREGAMENTO DO DATASET
# =============================================================================

print("\n" + "="*80)
print("CARREGAMENTO DO DATASET")
print("="*80)

# Download do dataset
path_kaggle = kagglehub.dataset_download("nicholasjhana/energy-consumption-generation-prices-and-weather")
print(f"Dataset baixado em: {path_kaggle}")

# Carregamento dos dataframes
df_energy = pd.read_csv(path_kaggle + '/energy_dataset.csv')
df_weather = pd.read_csv(path_kaggle + '/weather_features.csv')

print(f"\nDataset de Energia: {df_energy.shape[0]} linhas x {df_energy.shape[1]} colunas")
print(f"Dataset de Clima: {df_weather.shape[0]} linhas x {df_weather.shape[1]} colunas")

# =============================================================================
# 3. ANÁLISE EXPLORATÓRIA DOS DADOS
# =============================================================================

print("\n" + "="*80)
print("ANÁLISE EXPLORATÓRIA DOS DADOS")
print("="*80)

# 3.1 Conversão de datas
df_energy['time'] = pd.to_datetime(df_energy['time'])
df_weather['dt_iso'] = pd.to_datetime(df_weather['dt_iso'])

print("\n3.1 - Estrutura dos Dados")
print("-" * 80)
print("\nDataset de Energia:")
print(df_energy.info())
print("\nPrimeiras linhas:")
print(df_energy.head())

print("\n\nDataset de Clima:")
print(df_weather.info())
print("\nPrimeiras linhas:")
print(df_weather.head())

# 3.2 Estatísticas Descritivas
print("\n3.2 - Estatísticas Descritivas")
print("-" * 80)
print("\nDataset de Energia:")
print(df_energy.describe())

print("\nDataset de Clima:")
print(df_weather.describe())

# 3.3 Análise de Valores Faltantes
print("\n3.3 - Análise de Valores Faltantes")
print("-" * 80)

missing_energy = df_energy.isnull().sum()
missing_energy_pct = (missing_energy / len(df_energy)) * 100
missing_df_energy = pd.DataFrame({
    'Coluna': missing_energy.index,
    'Valores Faltantes': missing_energy.values,
    'Percentual (%)': missing_energy_pct.values
}).sort_values('Valores Faltantes', ascending=False)

print("\nDataset de Energia:")
print(missing_df_energy[missing_df_energy['Valores Faltantes'] > 0])

missing_weather = df_weather.isnull().sum()
missing_weather_pct = (missing_weather / len(df_weather)) * 100
missing_df_weather = pd.DataFrame({
    'Coluna': missing_weather.index,
    'Valores Faltantes': missing_weather.values,
    'Percentual (%)': missing_weather_pct.values
}).sort_values('Valores Faltantes', ascending=False)

print("\nDataset de Clima:")
print(missing_df_weather[missing_df_weather['Valores Faltantes'] > 0])

# 3.4 Visualizações - Série Temporal de Preços
print("\n3.4 - Análise Temporal")
print("-" * 80)

fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Série temporal completa
axes[0].plot(df_energy['time'], df_energy['price day ahead'], linewidth=0.5)
axes[0].set_title('Série Temporal de Preços de Energia (Completa)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Data')
axes[0].set_ylabel('Preço (€/MWh)')
axes[0].grid(True, alpha=0.3)

# Zoom em um período
df_sample = df_energy[df_energy['time'].dt.year == 2017].iloc[:30*24]
axes[1].plot(df_sample['time'], df_sample['price day ahead'])
axes[1].set_title('Série Temporal de Preços - Primeiro Mês de 2017', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Data')
axes[1].set_ylabel('Preço (€/MWh)')
axes[1].grid(True, alpha=0.3)

# Distribuição dos preços
axes[2].hist(df_energy['price day ahead'].dropna(), bins=50, edgecolor='black')
axes[2].set_title('Distribuição dos Preços de Energia', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Preço (€/MWh)')
axes[2].set_ylabel('Frequência')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('serie_temporal_precos.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo: serie_temporal_precos.png")
plt.show()

# Estatísticas da variável alvo
print("\nEstatísticas da Variável Alvo (price day ahead):")
print(f"Média: {df_energy['price day ahead'].mean():.2f} €/MWh")
print(f"Mediana: {df_energy['price day ahead'].median():.2f} €/MWh")
print(f"Desvio Padrão: {df_energy['price day ahead'].std():.2f} €/MWh")
print(f"Mínimo: {df_energy['price day ahead'].min():.2f} €/MWh")
print(f"Máximo: {df_energy['price day ahead'].max():.2f} €/MWh")

# 3.5 Análise de Sazonalidade
print("\n3.5 - Análise de Sazonalidade")
print("-" * 80)

# Criar features temporais
df_energy['hour'] = df_energy['time'].dt.hour
df_energy['day_of_week'] = df_energy['time'].dt.dayofweek
df_energy['month'] = df_energy['time'].dt.month

# Padrão por hora do dia
hourly_avg = df_energy.groupby('hour')['price day ahead'].mean()
daily_avg = df_energy.groupby('day_of_week')['price day ahead'].mean()
monthly_avg = df_energy.groupby('month')['price day ahead'].mean()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(hourly_avg.index, hourly_avg.values, marker='o')
axes[0].set_title('Preço Médio por Hora do Dia', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Hora')
axes[0].set_ylabel('Preço Médio (€/MWh)')
axes[0].grid(True, alpha=0.3)

axes[1].bar(daily_avg.index, daily_avg.values)
axes[1].set_title('Preço Médio por Dia da Semana', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Dia (0=Segunda, 6=Domingo)')
axes[1].set_ylabel('Preço Médio (€/MWh)')
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].plot(monthly_avg.index, monthly_avg.values, marker='o')
axes[2].set_title('Preço Médio por Mês', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Mês')
axes[2].set_ylabel('Preço Médio (€/MWh)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sazonalidade_precos.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo: sazonalidade_precos.png")
plt.show()

# 3.6 Análise de Correlações
print("\n3.6 - Análise de Correlações")
print("-" * 80)

# Selecionar colunas numéricas relevantes
numeric_cols = df_energy.select_dtypes(include=[np.number]).columns.tolist()
# Remover colunas desnecessárias
cols_to_remove = ['hour', 'day_of_week', 'month']
numeric_cols = [col for col in numeric_cols if col not in cols_to_remove]

# Calcular correlação com o preço
correlations = df_energy[numeric_cols].corr()['price day ahead'].sort_values(ascending=False)
print("\nTop 10 variáveis mais correlacionadas com o preço:")
print(correlations.head(10))

print("\nTop 10 variáveis menos correlacionadas (mais negativas) com o preço:")
print(correlations.tail(10))

# Matriz de correlação (top variáveis)
top_vars = correlations.abs().sort_values(ascending=False).head(15).index
corr_matrix = df_energy[top_vars].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação - Top 15 Variáveis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_correlacao.png', dpi=300, bbox_inches='tight')
print("\nGráfico salvo: matriz_correlacao.png")
plt.show()

# 3.7 Análise de Geração vs Preço
print("\n3.7 - Análise de Geração vs Preço")
print("-" * 80)

# Agregar geração total por tipo
df_energy['generation_fossil_total'] = (
    df_energy['generation fossil brown coal/lignite'] +
    df_energy['generation fossil gas'] +
    df_energy['generation fossil hard coal'] +
    df_energy['generation fossil oil']
).fillna(0)

df_energy['generation_renewable_total'] = (
    df_energy['generation solar'] +
    df_energy['generation wind onshore'] +
    df_energy['generation wind offshore'] +
    df_energy['generation hydro run-of-river and poundage']
).fillna(0)

# Scatter plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Total Load Actual vs Price
axes[0, 0].scatter(df_energy['total load actual'], df_energy['price day ahead'], 
                   alpha=0.3, s=1)
axes[0, 0].set_title('Carga Total vs Preço', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Carga Total (MW)')
axes[0, 0].set_ylabel('Preço (€/MWh)')
axes[0, 0].grid(True, alpha=0.3)

# Geração Fóssil vs Price
axes[0, 1].scatter(df_energy['generation_fossil_total'], df_energy['price day ahead'], 
                   alpha=0.3, s=1, color='brown')
axes[0, 1].set_title('Geração Fóssil vs Preço', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Geração Fóssil (MW)')
axes[0, 1].set_ylabel('Preço (€/MWh)')
axes[0, 1].grid(True, alpha=0.3)

# Geração Renovável vs Price
axes[1, 0].scatter(df_energy['generation_renewable_total'], df_energy['price day ahead'], 
                   alpha=0.3, s=1, color='green')
axes[1, 0].set_title('Geração Renovável vs Preço', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Geração Renovável (MW)')
axes[1, 0].set_ylabel('Preço (€/MWh)')
axes[1, 0].grid(True, alpha=0.3)

# Geração Nuclear vs Price
axes[1, 1].scatter(df_energy['generation nuclear'], df_energy['price day ahead'], 
                   alpha=0.3, s=1, color='purple')
axes[1, 1].set_title('Geração Nuclear vs Preço', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Geração Nuclear (MW)')
axes[1, 1].set_ylabel('Preço (€/MWh)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('geracao_vs_preco.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo: geracao_vs_preco.png")
plt.show()

# =============================================================================
# 4. PREPARAÇÃO DOS DADOS
# =============================================================================

print("\n" + "="*80)
print("PREPARAÇÃO DOS DADOS")
print("="*80)

# 4.1 Agregação de dados de clima por timestamp
print("\n4.1 - Agregação de Dados Climáticos")
print("-" * 80)

# Agregar clima por timestamp (média das cidades)
weather_agg = df_weather.groupby('dt_iso').agg({
    'temp': 'mean',
    'temp_min': 'mean',
    'temp_max': 'mean',
    'pressure': 'mean',
    'humidity': 'mean',
    'wind_speed': 'mean',
    'rain_1h': 'sum',
    'rain_3h': 'sum',
    'clouds_all': 'mean'
}).reset_index()

weather_agg.columns = ['time'] + ['weather_' + col for col in weather_agg.columns[1:]]

print(f"Dados climáticos agregados: {weather_agg.shape}")

# 4.2 Merge dos datasets
print("\n4.2 - Merge dos Datasets")
print("-" * 80)

df = df_energy.merge(weather_agg, on='time', how='left')
print(f"Dataset combinado: {df.shape[0]} linhas x {df.shape[1]} colunas")

# 4.3 Tratamento de Valores Faltantes
print("\n4.3 - Tratamento de Valores Faltantes")
print("-" * 80)

# Imputação por interpolação linear para séries temporais
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        missing_before = df[col].isnull().sum()
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        missing_after = df[col].isnull().sum()
        print(f"{col}: {missing_before} → {missing_after} valores faltantes")

# Remover linhas com valores faltantes restantes
df = df.dropna()
print(f"\nDataset após tratamento: {df.shape[0]} linhas")

# 4.4 Engenharia de Features
print("\n4.4 - Engenharia de Features")
print("-" * 80)

# Features temporais já criadas (hour, day_of_week, month)
df['day_of_month'] = df['time'].dt.day
df['quarter'] = df['time'].dt.quarter
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Features cíclicas
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Features de lag
df = df.sort_values('time')
df['price_lag_1'] = df['price day ahead'].shift(1)
df['price_lag_24'] = df['price day ahead'].shift(24)
df['price_lag_168'] = df['price day ahead'].shift(168)

# Médias móveis
df['price_ma_24'] = df['price day ahead'].rolling(window=24, min_periods=1).mean()
df['price_ma_168'] = df['price day ahead'].rolling(window=168, min_periods=1).mean()
df['load_ma_24'] = df['total load actual'].rolling(window=24, min_periods=1).mean()

# Feature de interação
df['load_generation_ratio'] = df['total load actual'] / (df['generation_fossil_total'] + 
                                                           df['generation_renewable_total'] + 
                                                           df['generation nuclear'] + 1)

# Remover linhas com NaN criados pelos lags
df = df.dropna()
print(f"Dataset após engenharia de features: {df.shape[0]} linhas x {df.shape[1]} colunas")

print("\nNovas features criadas:")
new_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                'price_lag_1', 'price_lag_24', 'price_lag_168',
                'price_ma_24', 'price_ma_168', 'load_ma_24', 'load_generation_ratio']
for feat in new_features:
    print(f"  - {feat}")

# 4.5 Seleção de Features
print("\n4.5 - Seleção de Features")
print("-" * 80)

# Remover colunas não necessárias
cols_to_drop = ['time', 'preco_energia', 'generation fossil coal-derived gas',
                'generation fossil oil shale', 'generation fossil peat',
                'generation geothermal', 'generation marine', 'generation other',
                'generation waste', 'forecast solar day ahead',
                'forecast wind offshore eday ahead', 'forecast wind onshore day ahead']

cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df = df.drop(columns=cols_to_drop)

# Identificar multicolinearidade
corr_matrix_full = df.select_dtypes(include=[np.number]).corr()
high_corr = []
for i in range(len(corr_matrix_full.columns)):
    for j in range(i+1, len(corr_matrix_full.columns)):
        if abs(corr_matrix_full.iloc[i, j]) > 0.95:
            col_i = corr_matrix_full.columns[i]
            col_j = corr_matrix_full.columns[j]
            if col_i != 'price day ahead' and col_j != 'price day ahead':
                high_corr.append((col_i, col_j, corr_matrix_full.iloc[i, j]))

if high_corr:
    print("\nPares de variáveis com correlação > 0.95:")
    for col1, col2, corr_val in high_corr[:10]:
        print(f"  {col1} <-> {col2}: {corr_val:.3f}")

# 4.6 Divisão dos Dados
print("\n4.6 - Divisão dos Dados")
print("-" * 80)

# Separar features e target
X = df.drop(columns=['price day ahead'])
y = df['price day ahead']

# Split temporal: 70% treino, 15% validação, 15% teste
n = len(df)
train_size = int(0.7 * n)
val_size = int(0.15 * n)

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_val = X.iloc[train_size:train_size+val_size]
y_val = y.iloc[train_size:train_size+val_size]

X_test = X.iloc[train_size+val_size:]
y_test = y.iloc[train_size+val_size:]

print(f"Conjunto de Treino: {X_train.shape[0]} amostras")
print(f"Conjunto de Validação: {X_val.shape[0]} amostras")
print(f"Conjunto de Teste: {X_test.shape[0]} amostras")
print(f"Total de features: {X_train.shape[1]}")

# 4.7 Normalização
print("\n4.7 - Normalização")
print("-" * 80)

# Identificar features cíclicas (não normalizar)
cyclic_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
features_to_scale = [col for col in X_train.columns if col not in cyclic_features]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
X_val_scaled[features_to_scale] = scaler.transform(X_val[features_to_scale])
X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])

print(f"Features normalizadas: {len(features_to_scale)}")
print(f"Features cíclicas (não normalizadas): {len(cyclic_features)}")

# =============================================================================
# 5. MODELAGEM
# =============================================================================

print("\n" + "="*80)
print("MODELAGEM")
print("="*80)

# Dicionário para armazenar resultados
results = {}

# 5.1 Modelo Baseline - Regressão Linear com Ridge
print("\n5.1 - Regressão Linear (Ridge)")
print("-" * 80)

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)

y_pred_ridge_val = ridge.predict(X_val_scaled)
y_pred_ridge_test = ridge.predict(X_test_scaled)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge_test))
r2_ridge = r2_score(y_test, y_pred_ridge_test)
mape_ridge = np.mean(np.abs((y_test - y_pred_ridge_test) / y_test)) * 100

results['Ridge'] = {
    'MAE': mae_ridge,
    'RMSE': rmse_ridge,
    'R2': r2_ridge,
    'MAPE': mape_ridge,
    'predictions': y_pred_ridge_test
}

print(f"MAE: {mae_ridge:.2f}")
print(f"RMSE: {rmse_ridge:.2f}")
print(f"R²: {r2_ridge:.4f}")
print(f"MAPE: {mape_ridge:.2f}%")

# 5.2 Random Forest
print("\n5.2 - Random Forest")
print("-" * 80)

print("Treinando modelo base...")
rf_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train_scaled, y_train)

y_pred_rf_val = rf_base.predict(X_val_scaled)
mae_val = mean_absolute_error(y_val, y_pred_rf_val)
print(f"MAE no conjunto de validação (modelo base): {mae_val:.2f}")

print("\nOtimizando hiperparâmetros com Grid Search...")
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [20, 30, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

tscv = TimeSeriesSplit(n_splits=3)
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                       param_grid_rf, cv=tscv, scoring='neg_mean_absolute_error',
                       verbose=1, n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)

print(f"\nMelhores hiperparâmetros: {grid_rf.best_params_}")

rf_best = grid_rf.best_estimator_
y_pred_rf_test = rf_best.predict(X_test_scaled)

mae_rf = mean_absolute_error(y_test, y_pred_rf_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
r2_rf = r2_score(y_test, y_pred_rf_test)
mape_rf = np.mean(np.abs((y_test - y_pred_rf_test) / y_test)) * 100

results['Random Forest'] = {
    'MAE': mae_rf,
    'RMSE': rmse_rf,
    'R2': r2_rf,
    'MAPE': mape_rf,
    'predictions': y_pred_rf_test,
    'model': rf_best
}

print(f"\nResultados no conjunto de teste:")
print(f"MAE: {mae_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R²: {r2_rf:.4f}")
print(f"MAPE: {mape_rf:.2f}%")

# 5.3 Gradient Boosting
print("\n5.3 - Gradient Boosting")
print("-" * 80)

print("Treinando modelo base...")
gb_base = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_base.fit(X_train_scaled, y_train)

y_pred_gb_val = gb_base.predict(X_val_scaled)
mae_val_gb = mean_absolute_error(y_val, y_pred_gb_val)
print(f"MAE no conjunto de validação (modelo base): {mae_val_gb:.2f}")

print("\nOtimizando hiperparâmetros com Grid Search...")
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}

grid_gb = GridSearchCV(GradientBoostingRegressor(random_state=42),
                       param_grid_gb, cv=tscv, scoring='neg_mean_absolute_error',
                       verbose=1, n_jobs=-1)
grid_gb.fit(X_train_scaled, y_train)

print(f"\nMelhores hiperparâmetros: {grid_gb.best_params_}")

gb_best = grid_gb.best_estimator_
y_pred_gb_test = gb_best.predict(X_test_scaled)

mae_gb = mean_absolute_error(y_test, y_pred_gb_test)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb_test))
r2_gb = r2_score(y_test, y_pred_gb_test)
mape_gb = np.mean(np.abs((y_test - y_pred_gb_test) / y_test)) * 100

results['Gradient Boosting'] = {
    'MAE': mae_gb,
    'RMSE': rmse_gb,
    'R2': r2_gb,
    'MAPE': mape_gb,
    'predictions': y_pred_gb_test,
    'model': gb_best
}

print(f"\nResultados no conjunto de teste:")
print(f"MAE: {mae_gb:.2f}")
print(f"RMSE: {rmse_gb:.2f}")
print(f"R²: {r2_gb:.4f}")
print(f"MAPE: {mape_gb:.2f}%")

# =============================================================================
# 6. AVALIAÇÃO E COMPARAÇÃO DOS MODELOS