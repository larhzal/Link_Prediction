import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def preprocess_nodes(input_file='json_files/reduced_nodes_connected.json', output_file='json_files/preprocessed_nodes.json', categorical_cols=None):
    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Select ID column to keep
    id_col = df['id']
    df = df.drop(columns=['id'])

    # Identify categorical columns if not provided
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Separate numerical columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Handle missing values (numerical): Fill with column mean
    imputer = SimpleImputer(strategy='mean')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # Normalize numerical values
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Affichage types des colonnes catégorielles (debug)
    print(df[categorical_cols].head())
    print(df[categorical_cols].apply(lambda col: col.map(type)).drop_duplicates())

    # Convert lists to strings in categorical columns (important)
    for col in categorical_cols:
        df[col] = df[col].apply(lambda x: ','.join(x) if isinstance(x, list) else x)

    # One-hot encode categorical columns
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(df[categorical_cols])
        cat_cols = encoder.get_feature_names_out(categorical_cols)
        df_cat = pd.DataFrame(cat_encoded, columns=cat_cols)
        df = df.drop(columns=categorical_cols).reset_index(drop=True)
        df = pd.concat([df, df_cat], axis=1)

    # Re-add ID column
    df.insert(0, 'id', id_col)

    # Convert back to list of dicts
    cleaned_data = df.to_dict(orient='records')

    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2)

    print(f"✅ Data preprocessed and saved to '{output_file}'.")

# Exemple d'exécution
if __name__ == "__main__":
    preprocess_nodes()
