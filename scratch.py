import os
from pathlib import Path

from global_params import BASEPATH
from Models.Preprocessing import all_states
from Utils import searching_all_files


for state in all_states:
    # dir_path = Path(BASEPATH) / Path(f'Outputs/Models/Performances/Baselines/{state}')
    # dir_path.mkdir(parents=False, exist_ok=True)
    dir_path = Path(BASEPATH) / Path(f'Outputs/Models/Performances/Baselines/{state}/Images')
    dir_path.mkdir(parents=False, exist_ok=True)
    print(dir_path.exists())


dir_path = Path(BASEPATH) / Path('Outputs/Models/Performances/Baselines/' )
all_files = searching_all_files(dir_path)
logic = []
for i in range(len(all_files)):
    file = str(all_files[i]).split('/')[-1]
    state = file.split('.')[0].split('_')[0]
    logic.append(int((dir_path / state).exists()))
    source = dir_path / file
    # print(dir_path)
    if state == 'Alabama':
        if file.split('.')[1] == 'csv':
            destination = dir_path / f"{state}/{file}"
            print(str(source) + "->"+ str(destination))
            if not destination.exists():
                source.replace(destination)
        if file.split('.')[1] == 'jpg':
            destination = dir_path / f"{state}/Images/{file}"
            print(str(source) + "->"+ str(destination))
            if not destination.exists():
                source.replace(destination)

dir_path = Path(BASEPATH) / Path(f'Outputs/Models/Performances/Baselines/')
all_files = searching_all_files(dir_path)
for f in all_files:
    os.remove(f)
# dir_path.mkdir(parents=False, exist_ok=True)
# print(dir_path.exists())
