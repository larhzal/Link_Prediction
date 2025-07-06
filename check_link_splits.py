import pandas as pd

def count_links(file_path, split_name):
    df = pd.read_csv(file_path)
    pos = df[df['label'] == 1].shape[0]
    neg = df[df['label'] == 0].shape[0]
    total = df.shape[0]
    print(f"🔍 {split_name}:")
    print(f"  ➕ Liens positifs : {pos}")
    print(f"  ➖ Liens négatifs : {neg}")
    print(f"  🔢 Total : {total}\n")

if __name__ == "__main__":
    count_links("train_links.csv", "Entraînement")
    count_links("val_links.csv", "Validation")
    count_links("test_links.csv", "Test")
