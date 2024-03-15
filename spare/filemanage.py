import os
import shutil

import yaml
import pandas as pd


class OverManage():

    def __init__(self, config_file:str='config.yml') -> None:
        # extract location of output folder
        # create if not exist
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        self.folder = self.config['output']['folder']
        os.makedirs(self.folder, exist_ok=True)

        self._extract_runs()

    
    def _runs_to_csv(self) -> None:
        self.runs_df.to_csv(f'{self.folder}/runs.csv', index_label='id')
    
    def _extract_runs(self) -> None:
        runs_filepath = f'{self.folder}/runs.csv'

        if not os.path.isfile(runs_filepath):
            self.runs_df = pd.DataFrame(columns=['name', 'num_obj'])
            self._runs_to_csv()
        else:
            self.runs_df = pd.read_csv(runs_filepath, index_col='id')


    def add_run(self, name:str, num_obj:int) -> None:
        if self.runs_df.shape[0] == 0:
            id = 0
        else:
            id = self.runs_df.index[-1] + 1

        new_run = pd.DataFrame({'name': [name], 'num_obj': [num_obj]}, [id])
        self.runs_df = pd.concat([self.runs_df, new_run])

        self.runs_df.sort_index()
        self._runs_to_csv()

        os.makedirs(f'{id}_{name}')


    def run_folder(self, run_id:int) -> str:
        return f"{self.folder}/{run_id}/{self.runs_df.loc[run_id]['name']}"

    
    def delete_run(self, run_id:int) -> None:
        to_delete = f"{run_id}_{self.runs_df.loc[run_id]['name']}"
        try:
            self.runs_df.drop(run_id, inplace=True)
        except:
            raise Exception(f'Run {run_id} did not exist')
        
        shutil.rmtree(to_delete)
        self._runs_to_csv()

    def delete_all_runs(self) -> None:
        sure = input('Type y if sure: ')
        if sure == 'y':
            for run_id in self.runs_df.index:
                self.delete_run(run_id)

