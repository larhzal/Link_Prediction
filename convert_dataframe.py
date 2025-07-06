import json
import pandas as pd

# Load the nodes JSON file
with open("json_files/reduced_nodes_connected.json") as f:
    node_data = json.load(f)

# Convert to DataFrame
nodes_df = pd.DataFrame(node_data)
print(nodes_df.columns)
print(nodes_df.head(2))
