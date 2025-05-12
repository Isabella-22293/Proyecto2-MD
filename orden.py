import os
import pandas as pd

# 1. Carga de datos
base = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base, "Dataset")
orders    = pd.read_csv(os.path.join(data_path, "olist_orders_dataset.csv"))
customers = pd.read_csv(os.path.join(data_path, "olist_customers_dataset.csv"))
reviews   = pd.read_csv(os.path.join(data_path, "olist_order_reviews_dataset.csv"))

# 2. Merge
df = (
    orders
    .merge(customers, how='left', on='customer_id')
    .merge(reviews,   how='left', on='order_id')
)
print("Tras merge:", df.shape)

# 3. Limpieza básica
df.ffill(inplace=True)
df.drop_duplicates(inplace=True)
print("Tras limpieza:", df.shape)

# 4. Ingeniería de fechas
date_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date',
    'review_creation_date',
    'review_answer_timestamp'
]
for col in date_cols:
    if col in df:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f"{col}_year"]  = df[col].dt.year.fillna(0).astype(int)
        df[f"{col}_month"] = df[col].dt.month.fillna(0).astype(int)
        df[f"{col}_day"]   = df[col].dt.day.fillna(0).astype(int)
        df.drop(columns=[col], inplace=True)

# 5. Eliminar columnas irrelevantes
to_drop = [
    'review_comment_title',
    'review_comment_message',
    'customer_unique_id',
    'order_id',
    'customer_id',
    'review_id'
]
df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)

# 6. One-hot de baja cardinalidad
low_card = ['order_status', 'customer_state']
df = pd.get_dummies(df, columns=[c for c in low_card if c in df.columns], drop_first=True)

# 7. Guardar preprocesado
out_path = os.path.join(base, "dataset_preprocesado.csv")
df.to_csv(out_path, index=False)
print("Preprocesado guardado en:", out_path)
