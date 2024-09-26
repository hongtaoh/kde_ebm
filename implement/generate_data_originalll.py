"""
This script generates data from EBM. Biomarker distribution parameters come from
real_theta_phi.json 

The script also reformats the data required to run the kde_ebm package by UCL
"""


from typing import List, Optional, Tuple, Dict
import json 
import pandas as pd 
import numpy as np 
import os 
import scipy.stats as stats

def generate_data_from_ebm(
    n_participants: int,
    S_ordering: List[str],
    real_theta_phi_file: str,
    healthy_ratio: float,
    output_dir: str,
    m, # combstr_m
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate an Event-Based Model (EBM) for disease progression.

    Args:
    n_participants (int): Number of participants.
    S_ordering (List[str]): Biomarker names ordered according to the order 
        in which each of them get affected by the disease.
    real_theta_phi_file (str): Directory of a JSON file which contains 
        theta and phi values for all biomarkers.
        See real_theta_phi.json for example format.
    output_dir (str): Directory where output files will be saved.
    healthy_ratio (float): Proportion of healthy participants out of n_participants.
    seed (Optional[int]): Seed for the random number generator for reproducibility.

    Returns:
    pd.DataFrame: A DataFrame with columns 'participant', "biomarker", 'measurement', 
        'diseased'.
    """
    # Parameter validation
    assert n_participants > 0, "Number of participants must be greater than 0."
    assert 0 <= healthy_ratio <= 1, "Healthy ratio must be between 0 and 1."

    # Set the seed for numpy's random number generator
    rng = np.random.default_rng(seed)

    # Load theta and phi values from the JSON file
    try:
        with open(real_theta_phi_file) as f:
            real_theta_phi = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {real_theta_phi} not fount")
    except json.JSONDecodeError:
        raise ValueError(f"File {real_theta_phi_file} is not a valid JSON file.")

    n_biomarkers = len(S_ordering)
    n_stages = n_biomarkers + 1

    n_healthy = int(n_participants * healthy_ratio)
    n_diseased = int(n_participants - n_healthy)

    # Generate disease stages
    kjs = np.concatenate((np.zeros(n_healthy, dtype=int), rng.integers(1, n_stages, n_diseased)))
    # shuffle so that it's not 0s first and then disease stages bur all random
    rng.shuffle(kjs)

    # Initiate biomarker measurement matrix (J participants x N biomarkers) with None
    X = np.full((n_participants, n_biomarkers), None, dtype=object)

    # Create distributions for each biomarker
    theta_dist = {biomarker: stats.norm(
        real_theta_phi[biomarker]['theta_mean'],
        real_theta_phi[biomarker]['theta_std']
    ) for biomarker in S_ordering}

    phi_dist = {biomarker: stats.norm(
        real_theta_phi[biomarker]['phi_mean'],
        real_theta_phi[biomarker]['phi_std']
    ) for biomarker in S_ordering}

    # Populate the matrix with biomarker measurements
    for j in range(n_participants):
        for n, biomarker in enumerate(S_ordering):
            # because for each j, we generate X[j, n] in the order of S_ordering,
            # the final dataset will have this ordering as well. 
            k_j = kjs[j]
            S_n = n + 1

            # Assign biomarker values based on the participant's disease stage
            # affected, or not_affected, is regarding the biomarker, not the participant
            if k_j >= 1:
                if k_j >= S_n:
                    # rvs() is affected by np.random()
                    X[j, n] = (
                        j, biomarker, theta_dist[biomarker].rvs(random_state=rng), k_j, S_n, 'affected')
                else:
                    X[j, n] = (j, biomarker, phi_dist[biomarker].rvs(random_state=rng),
                               k_j, S_n, 'not_affected')
            # if the participant is healthy
            else:
                X[j, n] = (j, biomarker, phi_dist[biomarker].rvs(random_state=rng),
                           k_j, S_n, 'not_affected')

    df = pd.DataFrame(X, columns=S_ordering)
    # make this dataframe wide to long
    df_long = df.melt(var_name="Biomarker", value_name="Value")
    data = df_long['Value'].apply(pd.Series)
    data.columns = ['participant', "biomarker", 'measurement', 'k_j', 'S_n', 'affected_or_not']

    biomarker_name_change_dic = dict(zip(S_ordering, range(1, n_biomarkers + 1)))
    data['diseased'] = data.apply(lambda row: row.k_j > 0, axis=1)
    data.drop(['k_j', 'S_n', 'affected_or_not'], axis=1, inplace=True)
    data['biomarker'] = data.apply(
        lambda row: f"{row.biomarker} ({biomarker_name_change_dic[row.biomarker]})", axis=1)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combination_str = f"{int(healthy_ratio*n_participants)}|{n_participants}_{m}"
    data.to_csv(f'{output_dir}/{combination_str}.csv', index=False)
    print("Data generation done! Output saved to:", combination_str)
    return data

def reformat_data(
        n, 
        r,
        m,
        output_dir
        ):
    # folder_name = "../kde_ebm/datasets/data/synthetic"
    comb_str = f"{int(n*r)}|{n}_{m}"
    df = pd.read_csv(f"{output_dir}/{comb_str}.csv")
    n_biomarkers = len(df.biomarker.unique())
    diseased_dic = dict(zip(df.participant, df.diseased))
    # from long to wide
    # columns will be like this: biomarker_1, biomarker_2, ...
    # note that participant is the index col
    dff = df.pivot(index=['participant'], columns='biomarker', values=['measurement'])
    # Define the desired order of the columns
    desired_order = df.biomarker.unique()
    # Reorder the DataFrame columns
    # level=1 to match the multi-index column level if using a pivot
    dff = dff.reindex(columns=desired_order, level=1) 
    dff['diseased'] = [int(diseased_dic[x]) for x in dff.index]
    new_file_dir = f"{output_dir}/{comb_str}_reformatted.csv"
    dff.to_csv(new_file_dir, index=False, header=None)    
    # Prepend the custom line to the file
    with open(new_file_dir, "r+") as file:
        content = file.read()  # Read the existing content
        file.seek(0, 0)  # Move the cursor to the beginning of the file
        file.write(f"{n},{n_biomarkers},CN,AD\n" + content)  # Write the new line and then the original content


if __name__ == "__main__":
    ns = [50, 200, 500]
    rs = [0.1, 0.25, 0.5, 0.75, 0.9]
    num_of_datasets_per_combination = 50
    # S_ordering =[
    #     'HIP-FCI', 'PCC-FCI', 'HIP-GMI', 'FUS-GMI', 'FUS-FCI'
    # ]
    S_ordering = np.array([
        'HIP-FCI', 'PCC-FCI', 'AB', 'P-Tau', 'MMSE', 'ADAS', 
        'HIP-GMI', 'AVLT-Sum', 'FUS-GMI', 'FUS-FCI'
    ])
    real_theta_phi_file = 'real_theta_phi.json'
    output_dir = "../kde_ebm/datasets/data/synthetic"

    for n in ns:
        for r in rs:
            for m in range(0, num_of_datasets_per_combination):

                generate_data_from_ebm(
                    n_participants=n,
                    S_ordering=S_ordering,
                    real_theta_phi_file=real_theta_phi_file,
                    healthy_ratio=r,
                    output_dir = output_dir,
                    m = m,
                    seed=m,
                )

                reformat_data(n, r, m, output_dir)




