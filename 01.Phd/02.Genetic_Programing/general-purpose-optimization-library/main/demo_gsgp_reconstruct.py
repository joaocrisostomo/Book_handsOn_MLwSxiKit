"""A simple demonstration of how to reconstruct a LISP-like tree created
by the efficient implementation of the GSGP algorithm when solving a
given SML problem. This example served as a basis for the paper.
"""


def main():
    import os
    import pandas as pd
    from gpol.utils.inductive_programming import prm_reconstruct_tree

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Demonstration of how to reconstruct a tree
    # Defines the processing device that was used by the GSGP solve algorithm
    device = "cpu"
    # Establishes the connection strings towards files containing the necessary pieces of information
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logFiles", "SML-IPOP_2021-04-19_10_0_40",
                        "reconstruct")
    path_history = os.path.join(path, "SML-IPOP_seed_0_history.csv")
    path_init_pop, path_rts = os.path.join(path, "init_pop"), os.path.join(path, "rts")
    # Reads the history file
    try:
        history = pd.read_csv(os.path.join(path_history), index_col=0)
    except FileNotFoundError:
        print('File not found! The connection string to the history file (specified in the parameter path_history) '
              'does not exist. Please verify the connection string', path_history)
    # Defines a reconstruction function
    reconstructor = prm_reconstruct_tree(history, path_init_pop, path_rts, device)
    # Chooses the most fit (training fitness) individual to be reconstructed
    start_idx = history["Fitness"].idxmin()
    print("Starting index (chosen individual):", start_idx)
    print("Individual's info:\n", history.loc[start_idx])
    # Reconstructs the individual
    ind = reconstructor(start_idx)
    print("Automatically reconstructed individual's representation:\n", ind)


if __name__ == "__main__":
    main()

