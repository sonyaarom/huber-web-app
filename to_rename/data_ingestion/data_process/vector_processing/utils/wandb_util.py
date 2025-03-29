import wandb
import pandas as pd

import wandb

def troubleshoot_wandb_run_access(entity, project, run_id):
    """
    Troubleshoot wandb run access issues
    
    Parameters:
    -----------
    entity : str
        Wandb entity (usually username or team name)
    project : str
        Project name
    run_id : str
        Specific run ID
    """
    try:
        # Authenticate first
        api = wandb.Api()
        
        # List all runs in the project to verify existence
        print("Listing runs in the project:")
        runs = api.runs(f"{entity}/{project}")
        
        print(f"Total runs found in {entity}/{project}: {len(runs)}")
        
        # Print run names and IDs for reference
        print("\nAvailable Run IDs:")
        for run in runs:
            print(f"Run Name: {run.name}, Run ID: {run.id}")
        
        # Try to access the specific run
        try:
            specific_run = api.run(f"{entity}/{project}/{run_id}")
            print(f"\nSuccessfully accessed run: {specific_run.name}")
        except ValueError as e:
            print(f"\nError accessing specific run: {e}")
            
    except Exception as e:
        print(f"Authentication or access error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Ensure you're logged in to wandb (`wandb login`)")
        print("2. Check your entity and project names")
        print("3. Verify the run ID exists")

# Example usage (replace with your details)
# troubleshoot_wandb_run_access('your-entity', 'your-project', 'your-run-id')

import wandb
import pandas as pd

def get_all_run_tables(entity, project, run_id):
    """
    Retrieve all tables for a specific run
    
    Parameters:
    -----------
    entity : str
        Wandb entity name
    project : str
        Project name
    run_id : str
        Run ID (can be partial or full)
    
    Returns:
    --------
    list of pd.DataFrame
        List of tables found for the run
    """
    # Initialize wandb API
    api = wandb.Api()
    
    try:
        # Find the specific run
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Collect all table artifacts
        tables = []
        for artifact in run.logged_artifacts():
            if artifact.type == 'run_table':
                try:
                    # Download the artifact
                    artifact_dir = artifact.download()
                    
                    # Find all CSV files in the downloaded artifact
                    import glob
                    import os
                    csv_files = glob.glob(os.path.join(artifact_dir, '*.csv'))
                    
                    for csv_file in csv_files:
                        table_df = pd.read_csv(csv_file)
                        # Add metadata about the artifact
                        table_df['artifact_name'] = artifact.name
                        tables.append(table_df)
                    
                    print(f"Found {len(csv_files)} CSV files in artifact: {artifact.name}")
                
                except Exception as e:
                    print(f"Error processing artifact {artifact.name}: {e}")
        
        if not tables:
            print("No tables found. Checking all artifacts:")
            for artifact in run.logged_artifacts():
                print(f"Artifact: {artifact.name}, Type: {artifact.type}")
        
        return tables
    
    except ValueError as e:
        print(f"Error accessing run: {e}")
        return []

# Example usage
# tables = get_all_run_tables('your-entity', 'your-project', '26qqigj7')
# for table in tables:
#     print(table)s

def concatenate_wandb_tables(project_name, run_name):
    """
    Concatenate all tables from a specific wandb run
    
    Parameters:
    -----------
    project_name : str
        Name of the wandb project
    run_name : str
        Name of the specific run
    
    Returns:
    --------
    pd.DataFrame
        Concatenated DataFrame with all tables from the run
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Find the specific run
    run = api.run(f"{project_name}/{run_name}")
    
    # Collect all tables
    tables = []
    for artifact in run.logged_artifacts():
        if artifact.type == 'table':
            # Download the artifact
            table_path = artifact.download()
            
            # Read the table
            table_df = pd.read_csv(table_path)
            
            # Add run name as a column for tracking
            table_df['wandb_run_name'] = run_name
            
            tables.append(table_df)
    
    # Concatenate all tables
    if tables:
        concatenated_table = pd.concat(tables, ignore_index=True)
        return concatenated_table
    else:
        print(f"No tables found for run: {run_name}")
        return None

import wandb
import pandas as pd
import json
import os

def concatenate_run_tables(entity, project, run_id):
    """
    Concatenate all run tables for a specific run without storing locally
    
    Parameters:
    -----------
    entity : str
        Wandb entity name
    project : str
        Project name
    run_id : str
        Run ID
    
    Returns:
    --------
    pd.DataFrame
        Concatenated DataFrame of all tables
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Find the specific run
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # Collect all tables
    tables = []
    
    for artifact in run.logged_artifacts():
        if artifact.type == 'run_table':
            try:
                # Download artifact
                artifact_dir = artifact.download()
                
                # Find JSON files
                json_files = [f for f in os.listdir(artifact_dir) if f.endswith('.json')]
                
                for json_file in json_files:
                    file_path = os.path.join(artifact_dir, json_file)
                    
                    # Read JSON directly
                    with open(file_path, 'r') as f:
                        table_data = json.load(f)
                        
                    # Convert to DataFrame
                    table_df = pd.DataFrame(table_data)
                    
                    # Add artifact name for tracking
                    table_df['artifact_name'] = artifact.name
                    
                    tables.append(table_df)
                    
                    print(f"Processed table from: {json_file}")
            
            except Exception as e:
                print(f"Error processing artifact {artifact.name}: {e}")
    
    # Concatenate tables
    if tables:
        concatenated_table = pd.concat(tables, ignore_index=True)
        
        # Upload back to wandb
        run = wandb.init(
            entity=entity, 
            project=project, 
            id=run_id, 
            resume=True
        )
        
        # Create a temporary file to upload
        temp_csv = 'concatenated_tables.csv'
        concatenated_table.to_csv(temp_csv, index=False)
        
        # Log as a new artifact
        artifact = wandb.Artifact('concatenated_text_comparison', type='run_table')
        artifact.add_file(temp_csv)
        run.log_artifact(artifact)
        
        # Clean up temporary file
        os.remove(temp_csv)
        
        print("Concatenated table uploaded successfully!")
        
        return concatenated_table
    else:
        print("No tables found to concatenate.")
        return None

# Example usage
# concatenate_run_tables('your-entity', 'your-project', '26qqigj7')
"""

Available Run IDs:
Run Name: llama_base_parent_[1, 3, 5], Run ID: 3a5klery
Run Name: llama_medium_parent_[1, 3, 5], Run ID: ux2m02iq
Run Name: llama_advanced_parent_[1, 3, 5], Run ID: tjmtcfwb
Run Name: llama_base_initial_[1, 3, 5], Run ID: 4paf19ol
Run Name: llama_medium_initial_[1, 3, 5], Run ID: iwrn75my
Run Name: llama_advanced_initial_[1, 3, 5], Run ID: daayfu1v
Run Name: openai_base_initial_[5], Run ID: xn4tjhrz
Run Name: openai_base_initial_[5], Run ID: k6m51xnu
Run Name: eval_20250312_232735, Run ID: ik4qd87e
Run Name: eval_20250312_233235, Run ID: x5ohfg5j
Run Name: eval_20250312_234354, Run ID: omqm3rn6
Run Name: eval_20250313_103138, Run ID: omr2tmwo
Run Name: eval_20250313_104850, Run ID: pxsdxwox
Run Name: eval_base_openai, Run ID: 3b9867b5
Run Name: eval_medium_openai, Run ID: 631m63g6
Run Name: eval_advanced_openai, Run ID: 8pzuond2
Run Name: eval_base_openai, Run ID: kxjhzccg
Run Name: eval_medium_openai, Run ID: 9l1bgyg3
Run Name: eval_advanced_openai, Run ID: gpy4b4wq
Run Name: eval_base_llama, Run ID: 26qqigj7
Run Name: eval_medium_llama, Run ID: ixsommja
Run Name: eval_advanced_llama, Run ID: rgakpvs3

"""
    
if __name__ == "__main__":
    project_name = "prompt-evaluation"
    run_name = "26qqigj7"
    entity = "konchakova-s-r-humboldt-universit-t-zu-berlin"
    concatenate_run_tables(entity, project_name, run_name)