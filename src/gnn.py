import pandas as pd
import torch
from torch_geometric.data import Data

def load_csv_to_pyg(path, node_feature_cols, edge_threshold=5.0, nrows=None):
    """
    Loads a CSV file containing node (residue) and edge (interaction) data
    and converts it into a PyTorch Geometric Data object.

    Parameters:
    - path: str, path to CSV file.
    - node_feature_cols: dict, mapping of feature names to encoding functions.
    - edge_threshold: float, distance threshold for edge connections (optional).
    - nrows: int, number of rows to load for quick testing (optional).

    Returns:
    - PyG Data object.
    """

    # Load CSV with optional row limit for testing
    df = pd.read_csv(path, nrows=nrows)

    print("\n CSV Loaded Successfully. First 5 Rows:")
    print(df.head())  # Debugging: Show the first few rows

    # Ensure Residues and Edges columns are properly parsed
    df["Residues"] = df["Residues"].apply(eval)  # Convert JSON strings to Python lists
    df["Edges"] = df["Edges"].apply(eval)

    #Extract Node Features
    node_features = []
    node_ids = []
    
    for index, row in df.iterrows():
        for residue in row["Residues"]:
            node_ids.append(residue["residue_id"])  # Store node IDs
            feature_vector = [
                residue["hydrophobicity"], 
                residue["charge"]
            ]
            node_features.append(feature_vector)

    if not node_features:
        raise ValueError("🚨 ERROR: No valid node features found. Check data format!")

    x = torch.tensor(node_features, dtype=torch.float)  # Convert to tensor

    print(f"\n✅ Node Feature Tensor Created: {x.shape}")  # Debugging: Check tensor shape

    #Process Edge List
    edge_list = []
    
    for index, row in df.iterrows():
        for edge in row["Edges"]:
            if edge["distance"] <= edge_threshold:  # Optional filtering by distance
                edge_list.append([int(edge["source"]), int(edge["target"])])

    if not edge_list:
        raise ValueError(" ERROR: No valid edges found. Check edge computation!")

    edge_index = torch.tensor(edge_list, dtype=torch.long).T  # Convert to PyTorch tensor (transpose for PyG)

    print(f"\n Edge Index Tensor Created: {edge_index.shape}")  # Debugging: Check tensor shape

    # 🔹 Step 3: Create PyG Graph Object
    graph_data = Data(x=x, edge_index=edge_index)

    print("\n🎯 PyTorch Geometric Data Object Created:")
    print(graph_data)

    return graph_data

# Load Data
csv_file = "parsed_metadata_with_residues.csv"

graph_data = load_csv_to_pyg(
    csv_file, 
    node_feature_cols={
        "hydrophobicity": lambda x: torch.tensor(x.values, dtype=torch.float),
        "charge": lambda x: torch.tensor(x.values, dtype=torch.float),
    },
    nrows=5  # Load only first 5 rows for testing
)

