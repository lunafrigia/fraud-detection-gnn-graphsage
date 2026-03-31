# Detección de Fraude Financiero con Graph Neural Networks (GraphSAGE)

Detección de fraude en transacciones e-commerce usando Graph Neural Networks. El modelo construye un grafo de 590K transacciones conectadas por tarjeta, email y dispositivo, y usa GraphSAGE para propagar señales de fraude a través de la red.

**Test AUC-ROC: 0.834** | **590K nodos** | **2.76M aristas** | **220 features**

---

## ¿Por qué GNN?

Los modelos tradicionales (XGBoost, LightGBM) analizan cada transacción de forma aislada. Pero el fraude tiene **estructura de red** — tarjetas compartidas, dispositivos reutilizados, cadenas de emails. Una GNN ve el grafo completo: si la cuenta A comparte tarjeta con B, y B comparte dispositivo con C que ya fue marcada como fraude, la GNN propaga esa señal hacia A y B aunque individualmente parezcan inocentes.

Esta es la misma arquitectura (GraphSAGE) que usan PayPal y Visa en producción para detección de fraude en tiempo real.

## Resultados

| Métrica | Valor |
|---------|-------|
| **Test AUC-ROC** | **0.834** |
| Val AUC-ROC | 0.852 |
| Test Accuracy | 95.4% |
| Precision (legítimo) | 97.7% |
| Recall (legítimo) | 97.5% |
| Precision (fraude) | 35.2% |
| Recall (fraude) | 37.4% |

**97.7% de precisión en transacciones legítimas** = muy pocas falsas alarmas bloqueando clientes reales.

### Contexto

| Método | AUC | Tipo |
|--------|-----|------|
| Baseline aleatorio | 0.500 | — |
| **GraphSAGE (este proyecto)** | **0.834** | **GNN** |
| XGBoost baseline | ~0.920 | Tabular (434 features) |
| LightGBM top Kaggle | ~0.960 | Tabular + feat. eng. extensivo |

En producción, GNN y modelos tabulares se combinan: los embeddings de GraphSAGE se convierten en features adicionales para XGBoost, capturando tanto patrones tabulares como estructura de red.

## Enfoque

```
Datos tabulares → Construir grafo de transacciones → GraphSAGE → Clasificación de fraude
                   (tarjeta/email/dispositivo compartido)  (3 capas)
```

### 1. Construcción del Grafo

Cada transacción es un **nodo**. Dos transacciones comparten una **arista** si tienen el mismo atributo de identidad:

| Atributo | Aristas | Lógica |
|----------|---------|--------|
| `card1` | 2,013,998 | Misma tarjeta de crédito |
| `card2` | 380,448 | Segundo identificador de tarjeta |
| `DeviceInfo` | 238,490 | Mismo dispositivo |
| `addr1` | 43,278 | Misma dirección de facturación |
| `P_emaildomain` | 40,572 | Mismo dominio de email comprador |
| `R_emaildomain` | 38,162 | Mismo dominio de email receptor |
| `card3` | 35,234 | Tercer identificador de tarjeta |
| `addr2` | 7,740 | Segunda dirección |
| **Total** | **2,756,332** | **4.7 vecinos/nodo promedio** |

### 2. Features por Nodo (220 por transacción)

- 25 numéricos: TransactionAmt, C1-C14, D1-D15, dist1-dist2
- 15 categóricos: ProductCD, tipo de tarjeta, dominios de email, dispositivo, M1-M9
- 180 features Vesta: V1-V339 (filtrados por <60% missing)

### 3. Arquitectura FraudSAGE

```
Input: 220 features por nodo

SAGEConv(220 → 128) + ReLU + Dropout(0.3)    ← vecinos directos (1 salto)
SAGEConv(128 → 128) + ReLU + Dropout(0.3)    ← vecinos de vecinos (2 saltos)
SAGEConv(128 → 64)  + ReLU                    ← 3 grados de separación
Linear(64 → 2)                                 ← fraude vs legítimo

Parámetros totales: 105,922
```

### 4. Entrenamiento

- Full-batch en GPU (el grafo ocupa 0.53 GB de 8 GB de VRAM)
- Split temporal: entrenar con pasado (70%), validar con intermedio (15%), testear con futuro (15%)
- CrossEntropyLoss con class weights (fraude pesa 27× más)
- Adam optimizer con ReduceLROnPlateau
- 80 épocas

## Dataset

**[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)** — Vesta Corporation / IEEE-CIS

- 590,540 transacciones e-commerce reales
- 3.5% tasa de fraude (20,663 fraudulentas)
- 434 features entre tablas de transacción e identidad
- Datos reales de la plataforma de procesamiento de pagos de Vesta

### Configuración

1. Descargar `train_transaction.csv` y `train_identity.csv` del link de Kaggle
2. Colocar ambos archivos en la misma carpeta que el notebook
3. Instalar dependencias:
```bash
pip install torch torch-geometric networkx scikit-learn pandas matplotlib seaborn tqdm
```
4. Ejecutar todas las celdas en orden

**Nota:** Requiere GPU con CUDA para tiempo de entrenamiento razonable. CPU funciona pero el entrenamiento será lento.

## Hallazgo Clave: Propagación de Señales de Red

El hallazgo más importante: los nodos con más vecinos fraudulentos reciben scores de fraude más altos del modelo. Esto demuestra que la GNN está propagando señales de la red, no solo usando features individuales — algo que ningún modelo tabular puede hacer.

## Stack Tecnológico

`Python` · `PyTorch` · `PyTorch Geometric` · `GraphSAGE` · `CUDA` · `NetworkX` · `Scikit-learn` · `Pandas` · `Matplotlib`

**Hardware:** NVIDIA RTX 5060 Laptop GPU (8 GB VRAM — 0.53 GB utilizados)

## Referencias

- Hamilton, W. et al. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS 2017.
- IEEE-CIS Fraud Detection Competition. Kaggle, 2019.
- Weber, M. et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with GCN*. KDD Workshop.
- AWS Blog (2021). *Real-time Fraud Detection using Graph Neural Networks*.

## Autor

**Mario Carvajal** — Economista aplicando Graph Neural Networks a problemas de riesgo financiero.

---

*Este proyecto es parte de una serie de portafolio aplicando Machine Learning a problemas económicos reales.*
