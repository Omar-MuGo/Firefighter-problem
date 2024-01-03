import glob
import ffp
import pandas
import tqdm
import numpy

def get_training_filenames(prefix="01"):
    filenames = glob.glob(f"instances\\{prefix}_training\\*")

    return filenames

def get_testing_filenames(prefix="01",byclass=None):
    filenames = glob.glob(f"instances\\{prefix}_testing\\*")

    if byclass != None:
        filenames = [ filename for filename in filenames if byclass in filename ]
    else:
        pass

    return filenames

classes =  [
    '50_ep0.1_',
    '50_ep0.15_',
    '50_ep0.2_',
    '50_ep0.25_',
    '50_r0.259_',
    '50_r0.299_',
    '50_r0.334_',
    '100_ep0.05_',
    '100_ep0.075_',
    '100_ep0.1_',
    '100_ep0.125_',
    '100_r0.169_',
    '100_r0.195_',
    '100_r0.219_',
    '500_ep0.015_',
    '500_ep0.02_',
    '500_ep0.025_',
    '500_r0.071_',
    '500_r0.083_',
    '500_r0.093_',
    '1000_ep0.0075_',
    '1000_ep0.01_',
    '1000_ep0.0125_',
    '1000_r0.05_',
    '1000_r0.058_',
    '1000_r0.065_'
]

hu_classes =  [
    '50_ep0.1_',
    '50_ep0.15_',
    '50_ep0.2_',
    '100_ep0.05_',
    '100_ep0.075_',
    '100_ep0.1_',
    '500_ep0.015_',
    '500_ep0.02_',
    '500_ep0.025_',
    '1000_ep0.0075_',
    '1000_ep0.01_',
    '1000_ep0.0125_',
]

def get_evaluation_table(models):
    collector = list()
    Hu_collector = list()

    # Evaluate the models in all instances
    group = list()
    for filename in tqdm.tqdm(get_testing_filenames(), desc="Evaluating all", ascii=" -"):
        row = dict()
        for name in models:
            problem = ffp.FFP(filename)
            row[name] = problem.solve(models[name], 1, False)
        group.append(row)

    collector.append(
        {
            "group" : "all",
            **pandas.DataFrame(group).mean()
        }
    )

    # Evaluate the models by instance class, and calculate the safe nodes for the Hu results comparison 
    for clazz in classes:
        number_of_nodes = int(clazz.split("_")[0])
        group = list()
        Hu_group = list()

        for filename in tqdm.tqdm(get_testing_filenames(byclass=clazz), desc="Evaluating class {:<15}".format(clazz), ascii=" -"):
            row = dict()
            Hu_row = dict()
            
            for name in models:
                problem = ffp.FFP(filename)
                row[name] = problem.solve(models[name], 1, False)

                # Safe nodes = (number of nodes) * ( 1 - %burning_nodes_feature )
                Hu_row[name] = round( number_of_nodes*(1-row[name]) )
            
            group.append(row)
            Hu_group.append(Hu_row)

        collector.append(
            {
                "group" : clazz,
                **pandas.DataFrame(group).mean()
            }
        )

        if clazz in hu_classes:
            Hu_collector.append(
                {
                    "group" : clazz,
                    **pandas.DataFrame(Hu_group).mean()
                }
            )
        else:
            pass
    
    # Create excel files
    pandas.DataFrame(collector).to_excel("_evaluation_table.xlsx")
    pandas.DataFrame(Hu_collector).to_excel("_Hu_comparable_table.xlsx")
    print("Created: _evaluation_table.xlsx")
    print("Created: _Hu_comparable_table.xlsx")