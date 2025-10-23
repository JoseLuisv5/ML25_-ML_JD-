import pandas as pd
from pathlib import Path
import os
import numpy as np

# Configuración de paths
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../datasets/customer_purchases/"

def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    return pd.read_csv(fullfilename)

def join_purchase_ids():
    print("🔄 Uniendo purchase_ids para train...")
    
    original_train = read_csv("customer_purchases_train")
    processed_train = read_csv("customer_purchases_train_final_preprocessed")
    
    print(f"📊 Original train shape: {original_train.shape}")
    print(f"📊 Processed train shape: {processed_train.shape}")
    
    # ✅ CORRECCIÓN: Manejar diferente número de filas
    # Las 500 filas extras son ejemplos negativos generados
    
    # Para las primeras 7289 filas (positivos), usar purchase_ids originales
    purchase_ids_original = original_train['purchase_id'].values
    
    # Para las 500 filas extras (negativos), generar nuevos purchase_ids
    # Usar números grandes para no confundir con los reales
    start_new_id = 100000
    new_purchase_ids = [f"SYNTH_{i}" for i in range(start_new_id, start_new_id + 500)]
    
    # Combinar purchase_ids
    all_purchase_ids = list(purchase_ids_original) + new_purchase_ids
    
    # Verificar que tenemos el mismo número
    if len(all_purchase_ids) != len(processed_train):
        print(f"⚠️  Ajustando purchase_ids...")
        all_purchase_ids = all_purchase_ids[:len(processed_train)]
    
    # Asignar purchase_ids al processed
    processed_with_ids = processed_train.copy()
    processed_with_ids['purchase_id'] = all_purchase_ids
    
    # Poner purchase_id al principio
    columns = processed_with_ids.columns.tolist()
    columns.remove('purchase_id')
    new_columns = ['purchase_id'] + columns
    processed_with_ids = processed_with_ids[new_columns]
    
    print(f"✅ Train con IDs shape: {processed_with_ids.shape}")
    print(f"✅ Purchase IDs únicos: {processed_with_ids['purchase_id'].nunique()}")
    
    # Guardar resultado
    output_path = DATA_DIR / "customer_purchases_train_with_ids.csv"
    processed_with_ids.to_csv(output_path, index=False)
    print(f"💾 Train con IDs guardado en: {output_path}")
    
    return processed_with_ids

def join_test_ids():
    print("\n🔄 Preparando datos de test...")
    
    original_test = read_csv("customer_purchases_test")
    processed_test = read_csv("customer_purchases_test_final_preprocessed")
    
    print(f"📊 Original test shape: {original_test.shape}")
    print(f"📊 Processed test shape: {processed_test.shape}")
    
    # Para test, usar purchase_ids originales directamente
    processed_test_with_ids = processed_test.copy()
    processed_test_with_ids['purchase_id'] = original_test['purchase_id'].values
    
    # Poner purchase_id al principio
    columns_test = processed_test_with_ids.columns.tolist()
    columns_test.remove('purchase_id')
    new_columns_test = ['purchase_id'] + columns_test
    processed_test_with_ids = processed_test_with_ids[new_columns_test]
    
    print(f"✅ Test con IDs shape: {processed_test_with_ids.shape}")
    print(f"✅ Purchase IDs únicos en test: {processed_test_with_ids['purchase_id'].nunique()}")
    
    # Guardar para uso interno
    output_path_internal = DATA_DIR / "customer_purchases_test_with_ids.csv"
    processed_test_with_ids.to_csv(output_path_internal, index=False)
    print(f"💾 Test con IDs guardado en: {output_path_internal}")
    
    return processed_test_with_ids

def create_kaggle_submission_template():
    """Crea el template para Kaggle con los purchase_ids del test original"""
    print("\n🎯 Creando template para Kaggle...")
    
    original_test = read_csv("customer_purchases_test")
    
    # Verificar que tenemos exactamente 978 purchase_ids
    purchase_ids = original_test['purchase_id'].unique()
    print(f"📊 Purchase IDs únicos en test: {len(purchase_ids)}")
    
    # Crear template para Kaggle
    kaggle_template = pd.DataFrame({
        'ID': purchase_ids,
        'pred': 0  # Placeholder
    })
    
    # Guardar template
    kaggle_output_path = DATA_DIR / "kaggle_submission_template.csv"
    kaggle_template.to_csv(kaggle_output_path, index=False)
    print(f"💾 Template para Kaggle guardado en: {kaggle_output_path}")
    
    return kaggle_template

if __name__ == "__main__":
    # Unir purchase_ids para train
    train_with_ids = join_purchase_ids()
    
    # Unir purchase_ids para test
    test_with_ids = join_test_ids()
    
    # Crear template para Kaggle
    kaggle_template = create_kaggle_submission_template()
    
    print("\n🎉 Proceso completado!")
    print(f"📁 Train con IDs: {len(train_with_ids)} filas, {train_with_ids['purchase_id'].nunique()} IDs únicos")
    print(f"📁 Test con IDs: {len(test_with_ids)} filas, {test_with_ids['purchase_id'].nunique()} IDs únicos")
    print(f"📁 Kaggle template: {len(kaggle_template)} purchase_ids")
    
    # Mostrar ejemplos
    print(f"\n📋 Ejemplo train (primeras 5 filas):")
    print(train_with_ids[['purchase_id']].head(5).to_string(index=False))
    
    print(f"\n📋 Ejemplo test (primeras 5 filas):")
    print(test_with_ids[['purchase_id']].head(5).to_string(index=False))
    
    print(f"\n📋 Ejemplo Kaggle template (primeras 5 filas):")
    print(kaggle_template.head(5).to_string(index=False))