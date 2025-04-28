import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Carga de datos
base = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base, "Dataset")
orders    = pd.read_csv(os.path.join(data_path, "olist_orders_dataset.csv"))
customers = pd.read_csv(os.path.join(data_path, "olist_customers_dataset.csv"))
reviews   = pd.read_csv(os.path.join(data_path, "olist_order_reviews_dataset.csv"))

#Merge
df = orders.merge(customers, how='left', on='customer_id') \
           .merge(reviews,  how='left', on='order_id')
print("Tras merge:", df.shape)

#Limpieza básica
df.ffill(inplace=True)
df.drop_duplicates(inplace=True)
print("Tras limpieza:", df.shape)

#Ingeniería de fechas (extraer año/mes/día)
for col in ['order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date',
            'review_creation_date',
            'review_answer_timestamp']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f"{col}_year"]  = df[col].dt.year.fillna(0).astype(int)
        df[f"{col}_month"] = df[col].dt.month.fillna(0).astype(int)
        df[f"{col}_day"]   = df[col].dt.day.fillna(0).astype(int)
        df.drop(columns=[col], inplace=True)

#Eliminar columnas irrelevantes o de muy alta cardinalidad
to_drop = [
    'review_comment_title',
    'review_comment_message',
    'customer_unique_id',
    'order_id',
    'customer_id',
    'review_id'
]
df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)

#Baja cardinalidad
low_card = ['order_status', 'customer_state']
for col in low_card:
    if col in df.columns:
        df[col] = df[col].astype('category')
df = pd.get_dummies(df, columns=[c for c in low_card if c in df.columns], drop_first=True)

# 7. LabelEncode de seller_id o product_id si en el futuro los incorporas
# (ejemplo)
# if 'seller_id' in df.columns:
#     le = LabelEncoder()
#     df['seller_id_le'] = le.fit_transform(df['seller_id'])
#     df.drop(columns=['seller_id'], inplace=True)

#Escalado de la variable objetivo (si lo deseas)
if 'review_score' in df.columns:
    scaler = StandardScaler()
    df['review_score'] = scaler.fit_transform(df[['review_score']])

#Guardar
out_path = os.path.join(base, "dataset_preprocesado.csv")
df.to_csv(out_path, index=False)
print("Preprocesado guardado en:", out_path)
