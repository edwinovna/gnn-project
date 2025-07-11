import os
import csv
import re
import json
import numpy as np
from Bio.PDB import PDBParser
from multiprocessing import Pool

# Directory containing PDB files
pdb_directory = os.path.join("imgt")

# Output CSV file for parsed metadata
output_file = os.path.join("parsed_metadata.csv")

# Physicochemical properties
hydrophobicity_scale = {
    "ALA": 1.8, "VAL": 4.2, "ILE": 4.5, "LEU": 3.8, "PHE": 2.8,
    "CYS": 2.5, "MET": 1.9, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
    "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5
}
charge_scale = {
    "ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1, "HIS": 0  # Add others as needed
}

def parse_remark_5_extended(pdb_file):
    """Parse REMARK 5 lines with multiple HCHAIN or AGCHAIN values."""
    remark_5_data = []
    with open(pdb_file, "r") as file:
        for line in file:
            if line.startswith("REMARK   5"):
                # Extract multiple values for HCHAIN, LCHAIN, AGCHAIN
                hchains = re.findall(r"HCHAIN=([A-Za-z,]+)", line)
                lchains = re.findall(r"LCHAIN=([A-Za-z,]+)", line)
                agchains = re.findall(r"AGCHAIN=([A-Za-z,]+)", line)
                agtypes = re.findall(r"AGTYPE=([A-Za-z,]+)", line)

                # Handle comma-separated values
                hchains = hchains[0].split(",") if hchains else []
                lchains = lchains[0].split(",") if lchains else []
                agchains = agchains[0].split(",") if agchains else []
                agtypes = agtypes[0].split(",") if agtypes else []

                # Store the extracted information
                remark_5_data.append({
                    "HCHAINS": hchains,
                    "LCHAINS": lchains,
                    "AGCHAINS": agchains,
                    "AGTYPES": agtypes,
                })
    return remark_5_data

def extract_residue_data(pdb_file, chain_mapping):
    """Extract residue data with features."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_file)

    residues = []

    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id in chain_mapping["HCHAINS"] + chain_mapping["LCHAINS"] + chain_mapping["AGCHAINS"]:
                for residue in chain:
                    try:
                        residue_type = residue.resname
                        coords = [
                            atom.coord for atom in residue if atom.name == "CA"
                        ]
                        if not coords:
                            continue  # Skip residues without alpha-carbon

                        hydrophobicity = hydrophobicity_scale.get(residue_type, 0)
                        charge = charge_scale.get(residue_type, 0)

                        residue_id = residue.id[1] if isinstance(residue.id[1], int) else str(residue.id[1])

                        residues.append({
                            "chain_id": chain_id,
                            "residue_name": residue_type,
                            "residue_id": residue_id,
                            "coordinates": coords[0],  # Single alpha-carbon coordinate
                            "hydrophobicity": hydrophobicity,
                            "charge": charge
                        })
                    except Exception as e:
                        print(f"Error processing residue {residue.id}: {e}")

    return residues

def compute_edges(residues, threshold=5.0):
    """Compute edges based on spatial proximity."""
    edges = []
    for i in range(len(residues)):
        for j in range(i + 1, len(residues)):
            coord1 = np.array(residues[i]['coordinates'])
            coord2 = np.array(residues[j]['coordinates'])
            distance = np.linalg.norm(coord1 - coord2)
            if distance < threshold:
                edges.append({
                    "source": residues[i]['residue_id'],
                    "target": residues[j]['residue_id'],
                    "distance": distance
                })
    return edges

def process_pdb_file(pdb_file):
    """Process a single PDB file."""
    try:
        full_path = os.path.join(pdb_directory, pdb_file)
        remark_5_data = parse_remark_5_extended(full_path)

        results = []
        for entry in remark_5_data:
            if not (entry["HCHAINS"] or entry["LCHAINS"] or entry["AGCHAINS"]):
                continue

            chain_mapping = {
                "HCHAINS": entry["HCHAINS"],
                "LCHAINS": entry["LCHAINS"],
                "AGCHAINS": entry["AGCHAINS"],
            }
            residues = extract_residue_data(full_path, chain_mapping)
            if len(residues) == 0:
                continue

            # Convert coordinates from ndarray to list
            for residue in residues:
                residue["coordinates"] = residue["coordinates"].tolist()

            edges = compute_edges(residues)

            results.append({
                "file": pdb_file,
                "hchains": ",".join(entry["HCHAINS"]),
                "lchains": ",".join(entry["LCHAINS"]),
                "agchains": ",".join(entry["AGCHAINS"]),
                "agtypes": ",".join(entry["AGTYPES"]),
                "residues": residues,
                "edges": edges
            })
        return results
    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
        return None

if __name__ == "__main__":
    # Use multiprocessing to parallelize file processing
    with Pool(processes=8) as pool:  # Adjust number of processes based on CPU cores
        pdb_files = [f for f in os.listdir(pdb_directory) if f.endswith(".pdb")]
        all_results = pool.map(process_pdb_file, pdb_files)

    # Write results to CSV with JSON serialization
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["File", "HCHAINS", "LCHAINS", "AGCHAINS", "AGTYPES", "Residues", "Edges"])

        # Loop through all results
        for result in all_results:
            if result:
                for entry in result:
                    # Serialize residues and edges as JSON strings
                    residues_json = json.dumps(entry["residues"])
                    edges_json = json.dumps(entry["edges"])

                    writer.writerow([
                        entry["file"],
                        entry["hchains"],
                        entry["lchains"],
                        entry["agchains"],
                        entry["agtypes"],
                        residues_json,
                        edges_json
                    ])
