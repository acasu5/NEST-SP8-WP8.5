
"""
@author: Adriano Casu, University of Cagliari, e-mail: adriano.casu@unica.it

"""


import numpy as np
import pandas as pd
import pandapower as pp
import matplotlib.pyplot as plt
import datetime
import math
import copy
import warnings
warnings.filterwarnings("ignore")  

class OutageScenario():

    def __init__(self, grid_base, lines_df, buses_df, loads_df, generators_df, l_day_profile_df, l_week_profile_df, 
                 l_month_profile_df, l_year_profile_df, g_day_profile_df, g_week_profile_df, g_month_profile_df, 
                 g_year_profile_df):
        
        self.event = 'HW' # choose between 'HW' (heatwaves) or 'WS' (windstorms)
        self.failure_probability = 0.25
        self.n_crews = 8
        self.year = 2050 # choose a year between 2010 and 2050
        self.lv_mg = False
        self.mv_mg = True
        self.strg = True
        
        self.grid_base = grid_base
        self.lines_df = lines_df
        self.buses_df = buses_df
        self.loads_df = loads_df
        self.generators_df = generators_df
        self.grid = copy.deepcopy(self.grid_base)
        
        self.l_day_profile_df = l_day_profile_df
        self.l_week_profile_df = l_week_profile_df
        self.l_month_profile_df = l_month_profile_df
        self.l_year_profile_df = l_year_profile_df
        self.g_day_profile_df = g_day_profile_df
        self.g_week_profile_df = g_week_profile_df
        self.g_month_profile_df = g_month_profile_df
        self.g_year_profile_df = g_year_profile_df
        
        self.event_date, self.event_start, self.faults_df = self._scenario_extraction()
    
    def _event_occurrence(self):
        
        #set date limits
        if self.event == 'HW':
            first_date = datetime.datetime(self.year, 5, 1, 0, 0, 0)   # YYYY-05-01, 00:00:00
            last_date = datetime.datetime(self.year, 8, 31, 0, 0, 0)   # YYYY-08-31, 00:00:00
        else:
            first_date = datetime.datetime(self.year, 1, 1, 0, 0, 0)   # YYYY-05-01, 00:00:00
            last_date = datetime.datetime(self.year, 12, 31, 0, 0, 0)  # YYYY-12-31, 00:00:00
        
        #extract event date and time
        date = first_date + (last_date - first_date) * np.random.rand()
        time = date.hour + date.minute / 60
        
        return date, time
    
    def _fault_estimation(self):
            
            #extract_outage_scenario
            faulted_lines = []
            if self.event == 'HW':
                for index, row in self.lines_df.iterrows():
                    if self.grid.line.at[index, 'type'] == 'cs':
                        for j in range(row['N_joints']):
                            rnd_number = np.random.uniform(0,1)
                            if rnd_number <= self.failure_probability:
                                faulted_lines.append(self.grid.line.at[index, 'name'])
                        for t in range(row['N_terminals']):
                            rnd_number = np.random.uniform(0,1)
                            if rnd_number <= self.failure_probability * 0.5:
                                faulted_lines.append(self.grid.line.at[index, 'name'])
            elif self.event == 'HW':
                for index, row in self.lines_df.iterrows():
                    if self.grid.line.at[index, 'type'] == 'ol':
                            rnd_number = np.random.uniform(0,1)
                            if rnd_number <= self.failure_probability:
                                faulted_lines.append(self.grid.line.at[index, 'name'])
            faulted_lines = sorted(set(faulted_lines))
            
            return faulted_lines
    
    def _scenario_extraction(self):
        
        #extract times for each fault
        event_date, event_start = self._event_occurrence()
        components_list = self._fault_estimation()
        time_to_fault = []
        individuation_time = []
        time_to_repair = []
        for c in components_list:
            time_to_fault.append(1 + np.random.exponential())
            individuation_time.append(1 + np.random.weibull(3.5))
            time_to_repair.append(8 + np.random.exponential())

        df_faults = pd.DataFrame(components_list, columns=['Component'])
        df_faults = df_faults.merge(self.lines_df[['Line ID', 'From', 'To']], left_on='Component', right_on='Line ID', how='left')
        df_faults = df_faults.merge(self.buses_df[['Node ID', 'Feeder','Distance']], left_on='To', right_on='Node ID', how='left')
        df_faults = df_faults.drop(columns=['Line ID', 'Node ID'])
        df_faults['Start'] = event_start
        df_faults['Degradation'] = time_to_fault
        df_faults['Individuation'] = individuation_time
        df_faults['Restoration'] = time_to_repair
        
        return event_date, event_start, df_faults
    
    def _reset(self):

        self.faults_df['state'] = 'degrading'
        self.faults_df['pending_start_time'] = np.nan
        self.faults_df['pending_time_int'] = 0.0
        self.faults_df['occurrence_time'] = self.faults_df['Start'] + self.faults_df['Degradation']
        self.faults_df['visibility_time'] = np.nan
        self.faults_df['hidden_start_time'] = np.nan
        self.faults_df['hidden_time_int'] = 0.0
        self.faults_df['individuation_time'] = np.nan
        self.faults_df['assignment_time'] = np.nan
        self.faults_df['repair_end_time'] = np.nan
        self.faults_df['assigned_crew'] = None
        
        self.current_time = 0.0
        
        self.crews = [{'id': i, 'busy_until': 0.0, 'assigned_fault_idx': None} for i in range(self.n_crews)]
        
    def _update_fault_states(self):
        
        nxg = pp.topology.create_nxgraph(self.grid)
        fed_buses_idx = list(pp.topology.connected_component(nxg, 1))
        fed_buses = [self.buses_df.loc[idx-1, 'Node ID'] for idx in fed_buses_idx]
        
        for idx, row in self.faults_df.iterrows():
            
            if row['state'] in ['assigned', 'completed']:
                continue
            
            # degrading -> pending
            if self.event == 'HW' and row['state'] == 'degrading' and row['From'] not in fed_buses and row['To'] not in fed_buses:
                self.faults_df.loc[idx, 'state'] = 'pending'
                self.faults_df.loc[idx, 'pending_start_time'] = self.current_time

            # pending -> degrading
            if self.event == 'HW' and row['state'] == 'pending' and (row['From'] in fed_buses or row['To'] in fed_buses):
                self.faults_df.loc[idx, 'state'] = 'degrading'
                self.faults_df.loc[idx, 'pending_time_int'] += self.current_time - self.faults_df.loc[idx, 'pending_start_time']
                self.faults_df.loc[idx, 'occurrence_time'] += self.faults_df.loc[idx, 'pending_time_int']
            
            # degrading -> occurred
            if row['state'] == 'degrading' and self.current_time >= row['occurrence_time']:
                self.faults_df.loc[idx, 'state'] = 'occurred'

            # occurred -> visible
            if row['state'] == 'occurred' and (row['From'] in fed_buses or row['To'] in fed_buses):
                self.faults_df.loc[idx, 'state'] = 'visible'
                self.faults_df.loc[idx, 'visibility_time'] = self.current_time
                self.faults_df.loc[idx, 'individuation_time'] = self.faults_df.loc[idx, 'visibility_time'] + self.faults_df.loc[idx, 'Individuation']
                
            # visible -> hidden
            if row['state'] == 'visible' and row['From'] not in fed_buses and row['To'] not in fed_buses:
                self.faults_df.loc[idx, 'state'] = 'hidden'
                self.faults_df.loc[idx, 'hidden_start_time'] = self.current_time

            # hidden -> visible 
            if row['state'] == 'hidden' and (row['From'] in fed_buses or row['To'] in fed_buses):
                self.faults_df.loc[idx, 'state'] = 'visible'
                self.faults_df.loc[idx, 'hidden_time_int'] += self.current_time - self.faults_df.loc[idx, 'hidden_start_time']
                self.faults_df.loc[idx, 'individuation_time'] += self.faults_df.loc[idx, 'hidden_time_int']

            # visible -> reparable
            if row['state'] == 'visible' and self.current_time >= row['individuation_time']:
                self.faults_df.loc[idx, 'state'] = 'reparable'

    def _update_network_topology(self):

        self.grid.line['in_service'] = True

        for idx, row in self.faults_df.iterrows():
            if row['state'] in ['occurred', 'visible', 'hidden', 'reparable', 'assigned']:
                self.grid.line.loc[self.grid.line['name'] == row['Component'], 'in_service'] = False

    def _complete_repairs(self):

        for idx, row in self.faults_df.iterrows():            
            if row['state'] == 'assigned' and self.current_time >= row['repair_end_time']:               

                # assigned -> completed
                self.faults_df.loc[idx, 'state'] = 'completed'
                    
                if row['assigned_crew'] is not None:
                    self.crews[row['assigned_crew']]['assigned_fault_idx'] = None
                    self.crews[row['assigned_crew']]['busy_until'] = self.current_time

    def _assign_faults_to_crews(self):

        free_crews = [crew for crew in self.crews if crew['assigned_fault_idx'] is None]
        
        if not free_crews:
            return

        available_faults = []
        for idx, row in self.faults_df.iterrows():
            if row['state'] == 'reparable':
                    score = row['Distance']
                    available_faults.append((score, idx))
        
        if not available_faults:
            return

        available_faults.sort(key=lambda x: x[0])
        
        for crew, (score, fault_idx) in zip(free_crews, available_faults):

            # reparable -> assigned
            self.faults_df.at[fault_idx, 'state'] = 'assigned'
            self.faults_df.at[fault_idx, 'assignment_time'] = self.current_time
            self.faults_df.at[fault_idx, 'repair_end_time'] = self.current_time + self.faults_df.at[fault_idx, 'Restoration']
            self.faults_df.at[fault_idx, 'assigned_crew'] = crew['id']

            self.crews[crew['id']]['assigned_fault_idx'] = fault_idx
            self.crews[crew['id']]['busy_until'] = self.faults_df.at[fault_idx, 'repair_end_time']
            
    def _advance_time(self):
        events = []
        for idx, row in self.faults_df.iterrows():

            if row['state'] == 'degrading' and row['occurrence_time'] > self.current_time:
                events.append(row['occurrence_time'])
            
            if row['state'] == 'visible' and row['individuation_time'] > self.current_time:
                events.append(row['individuation_time'])
            
            if row['state'] == 'assigned' and row['repair_end_time'] > self.current_time:
                events.append(row['repair_end_time'])
            
        if events:
            self.current_time = min(events)
    
    def _step(self):
        
        if all(state == 'completed' for state in self.faults_df['state'].values):
            return True
        
        self._update_fault_states()
        self._update_network_topology()
        self._update_fault_states()
        self._complete_repairs()
        self._assign_faults_to_crews()
        self._update_network_topology()
        self._update_fault_states()
        self._advance_time()
        
        return False
   
    def _create_profiles(self):
    
        #create load and generator profiles
        daily_l_profiles = []
        daily_g_profiles = []
        date = self.event_date
        t_window = self.faults_df['repair_end_time'].max()  
        d = 0
        while True: 
            loadshapes = self.l_day_profile_df.copy()
            genshapes = self.g_day_profile_df.copy()
    
            weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            weekday = weekdays[date.weekday()]
            selected_weekday = self.l_week_profile_df[self.l_week_profile_df['Day'] == weekday].iloc[0]     
            common_columns = self.l_day_profile_df.columns.intersection(self.l_week_profile_df.columns)  
            loadshapes[common_columns] = self.l_day_profile_df[common_columns].multiply(selected_weekday[common_columns], axis=1)
    
            selected_weekday = self.g_week_profile_df[self.g_week_profile_df['Day'] == weekday].iloc[0]     
            common_columns = self.g_day_profile_df.columns.intersection(self.g_week_profile_df.columns)
            genshapes[common_columns] = self.g_day_profile_df[common_columns].multiply(selected_weekday[common_columns], axis=1)
    
            selected_month = self.l_month_profile_df.iloc[date.month - 1]
            common_columns = loadshapes.columns.intersection(self.l_month_profile_df.columns)
            loadshapes[common_columns] = loadshapes[common_columns].multiply(selected_month[common_columns], axis=1)
    
            selected_month = self.g_month_profile_df.iloc[date.month - 1]
            common_columns = genshapes.columns.intersection(self.g_month_profile_df.columns)
            genshapes[common_columns] = genshapes[common_columns].multiply(selected_month[common_columns], axis=1)
            
            selected_year = self.l_year_profile_df[self.l_year_profile_df['Year'] == date.year].iloc[0]
            common_columns = loadshapes.columns.intersection(self.l_year_profile_df.columns)
            loadshapes[common_columns] = loadshapes[common_columns].multiply(selected_year[common_columns], axis=1)
            
            selected_year = self.g_year_profile_df[self.g_year_profile_df['Year'] == date.year].iloc[0]
            common_columns = genshapes.columns.intersection(self.g_year_profile_df.columns)
            genshapes[common_columns] = genshapes[common_columns].multiply(selected_year[common_columns], axis=1)
            
            loadshapes['Time'] = [i * 0.25 + 24 * d for i in range(len(loadshapes))]
            genshapes['Time'] = [i * 0.25 + 24 * d for i in range(len(genshapes))] 
            
            if t_window > 24:
                loadshapes = loadshapes.iloc[:-1]
                genshapes = genshapes.iloc[:-1]
                daily_l_profiles.append(loadshapes)
                daily_g_profiles.append(genshapes)
                date = date + datetime.timedelta(days=1)
                t_window -= 24
                d += 1
            else:
                daily_l_profiles.append(loadshapes)
                daily_g_profiles.append(genshapes)
                break
        loadshapes = pd.concat(daily_l_profiles)
        genshapes = pd.concat(daily_g_profiles)
        loadshapes = loadshapes.reset_index(drop=True)
        genshapes = genshapes.reset_index(drop=True)
    
        dict_loads = {}
        dict_gens = {}
        for idx in self.loads_df.index:
            load_type = self.loads_df.loc[idx, 'Load Type']
            dict_loads[idx] = loadshapes[load_type]
        load_profiles = pd.concat(dict_loads, axis=1)
        load_profiles = load_profiles.copy()
        for idx in self.generators_df.index:
            gen_type = self.generators_df.loc[idx, 'Profile Type']
            dict_gens[idx] = genshapes[gen_type]
        gen_profiles = pd.concat(dict_gens, axis=1)
        gen_profiles = gen_profiles.copy() 
        load_profiles['Time'] = [i * 0.25 for i in range(len(load_profiles))]
        gen_profiles['Time'] = [i * 0.25 for i in range(len(gen_profiles))]
        
        return load_profiles, gen_profiles
    
    def _create_results(self):
        
        self.df_results = pd.DataFrame()
        
        self.df_results.at[0, 'Event'] = self.event
        self.df_results.at[0, 'Year'] = self.year
        self.df_results.at[0, 'Failure probability'] = self.failure_probability
        self.df_results.at[0, 'Crews'] = self.n_crews
        self.df_results.at[0, 'LV microgrids'] = self.lv_mg
        self.df_results.at[0, 'MV microgrids'] = self.mv_mg
        self.df_results.at[0, 'Storage'] = self.strg
        
    def _calculate_power_not_supplied(self):
        
        self.grid.line['in_service'] = True
        load_profiles, gen_profiles = self._create_profiles()
    
        self.lv_microgrids = self.grid.bus[['name']].copy()
        self.mv_microgrids = self.grid.bus[['name']].copy()
        df_temp = self.faults_df.copy()
        df_temp = df_temp.sort_values(by='occurrence_time')
        df_temp = df_temp.reset_index(drop=True)
        df_t_profile = pd.DataFrame(columns=['Time', 'Performance level'])
        df_p_profile = pd.DataFrame(columns=['Time', 'P_base', 'P_not_supplied'])
        df_t_profile.loc[0] = [0,1]
        df_p_profile.loc[0] = [0,0,0]
        t_start = (self.event_start // 0.25) * 0.25
        df_t_profile.loc[1] = [t_start, 1]
        df_p_profile.loc[1] = [t_start, 0, 0]
        
        current_step = ((df_temp.at[0, 'occurrence_time']) // 0.25) * 0.25
        df_microgrids = 0
        while current_step < (df_temp['repair_end_time'].max()// 0.25) * 0.25 + 0.25:
            next_step = current_step + 0.25
            
            for index, comp in df_temp.iterrows():
                 if ((comp['occurrence_time'] // 0.25) * 0.25) <= current_step and ((comp['repair_end_time'] // 0.25) * 0.25) > current_step:
                     self.grid.line.loc[self.grid.line['name'] == comp['Component'], 'in_service'] = False
                 else:
                     self.grid.line.loc[self.grid.line['name'] == comp['Component'], 'in_service'] = True
                          
            for index, load in self.grid_base.load.iterrows():
                self.grid_base.load.at[index, 'scaling'] = load_profiles.loc[load_profiles['Time'] == current_step, index].values[0]
                self.grid.load.at[index, 'scaling'] = load_profiles.loc[load_profiles['Time'] == current_step, index].values[0]
                
            for index, gen in self.grid_base.sgen.iterrows():
                self.grid_base.sgen.at[index, 'scaling'] = gen_profiles.loc[gen_profiles['Time'] == current_step, index].values[0]
                self.grid.sgen.at[index, 'scaling'] = gen_profiles.loc[gen_profiles['Time'] == current_step, index].values[0]
                
            self.grid_base.load['p_mw_res'] = self.grid_base.load['p_mw'] * self.grid_base.load['scaling']
            p_base = self.grid_base.load['p_mw_res'].sum()
            
            self.nxg = pp.topology.create_nxgraph(self.grid)
            self.safe_buses = list(pp.topology.connected_component(self.nxg, 1))
            
            self.grid.load['p_mw_res'] = (self.grid.load['p_mw'] * self.grid.load['scaling']).where(self.grid.load.bus.isin(self.safe_buses), 0)
            p_supplied = self.grid.load['p_mw_res'].sum()
            
            lv_mg_contribution = 0
            if self.lv_mg:
                lv_mg_contribution = self._low_voltage_microgrids()
            mv_mg_contribution = 0
            if self.mv_mg:
                mv_mg_contribution = self._medium_voltage_microgrids()
            
            df_t_profile.loc[len(df_p_profile)] = [current_step , 1 - (p_base - (p_supplied + lv_mg_contribution + mv_mg_contribution))/p_base]
            df_p_profile.loc[len(df_p_profile)] = [current_step , p_base, p_base - (p_supplied + lv_mg_contribution + mv_mg_contribution)]
            current_step = next_step
        
        df_t_profile.loc[-1] = [load_profiles['Time'].max(), 1]
        df_p_profile.loc[-1] = [load_profiles['Time'].max(), 0, 0]
        
        if self.lv_mg:
            df_microgrids = self.lv_microgrids
        elif self.mv_mg:
            df_microgrids = self.mv_microgrids
        
        self._create_results()
        
        return df_t_profile, df_p_profile, df_microgrids, self.df_results
    
    def _low_voltage_microgrids(self):
        
        self.grid.sgen['p_sgen_mg'] = (self.grid.sgen['p_mw'] * self.grid.sgen['scaling']).where(~self.grid.sgen.bus.isin(self.safe_buses), 0)
        p_mg_sgens = self.grid.sgen.groupby('bus', as_index=False)['p_sgen_mg'].sum()
        
        self.grid.load['p_load_mg'] = (self.grid.load['p_mw'] * self.grid.load['scaling']).where(~self.grid.load.bus.isin(self.safe_buses), 0)
        p_mg_loads = self.grid.load.groupby('bus', as_index=False)['p_load_mg'].sum()
        
        self.lv_microgrids['p_sgen_mg'] = self.lv_microgrids.index.map(p_mg_sgens.set_index('bus')['p_sgen_mg'])
        self.lv_microgrids['p_load_mg'] = self.lv_microgrids.index.map(p_mg_loads.set_index('bus')['p_load_mg'])
        self.lv_microgrids = self.lv_microgrids.fillna(0)
        
        if self.strg:
            if 'p_strg_a' not in self.lv_microgrids.columns:
                self.lv_microgrids['p_strg_b'] = 0
                self.lv_microgrids['p_strg_a'] = (self.lv_microgrids['p_sgen_mg'] - self.lv_microgrids['p_load_mg']).clip(lower=0)
                self.lv_microgrids['e_strg'] = 0
                self.lv_microgrids['e_strg_max'] = 0
                self.lv_microgrids['p_lv_mg'] = self.lv_microgrids[['p_sgen_mg', 'p_load_mg']].min(axis=1)
            else:
                self.lv_microgrids['p_strg_b'] = self.lv_microgrids['p_strg_a']
                self.lv_microgrids['p_lv_mg'] = np.where((self.lv_microgrids['p_load_mg'] - self.lv_microgrids['p_sgen_mg']) > self.lv_microgrids['p_strg_b'], 
                                                         self.lv_microgrids['p_sgen_mg'] + self.lv_microgrids['p_strg_b'], self.lv_microgrids['p_load_mg'])
                self.lv_microgrids['p_strg_a'] = np.where((self.lv_microgrids['p_load_mg'] - self.lv_microgrids['p_sgen_mg']) >= self.lv_microgrids['p_strg_b'],0,
                                                          self.lv_microgrids['p_strg_b'] + self.lv_microgrids['p_sgen_mg'] - self.lv_microgrids['p_load_mg'])
                self.lv_microgrids['e_strg'] = np.where((self.lv_microgrids['p_strg_b'] - self.lv_microgrids['p_strg_a']) > 0,
                                                        self.lv_microgrids['e_strg'] + (self.lv_microgrids['p_strg_b'] - self.lv_microgrids['p_strg_a']), 0) * 15/60
                self.lv_microgrids['e_strg_max'] = self.lv_microgrids[['e_strg_max', 'e_strg']].max(axis=1)
        else:
            self.lv_microgrids['p_lv_mg'] = self.lv_microgrids[['p_sgen_mg', 'p_load_mg']].min(axis=1)
        
        lv_mg_contribution = self.lv_microgrids['p_lv_mg'].sum()
        
        return lv_mg_contribution

    def _medium_voltage_microgrids(self):
        
        self.grid.sgen['weight'] = self.grid.sgen['p_mw'].where(~self.grid.sgen.bus.isin(self.safe_buses))
        p_mg_sgens = self.grid.sgen.groupby('bus', as_index=False)['weight'].sum()
        self.mv_microgrids['weight'] = self.mv_microgrids.index.map(p_mg_sgens.set_index('bus')['weight'])
        self.mv_microgrids = self.mv_microgrids.fillna(0)
        self.mv_microgrids['mg'] = pd.Series(dtype=object)
        self.mv_microgrids['p_sgen_mg'] = 0
        self.mv_microgrids['p_load_mg'] = 0
                
        for i, row in self.mv_microgrids.iterrows():
            if row['weight'] != 0:
                mg = tuple(sorted(pp.topology.connected_component(self.nxg, i)))
                self.mv_microgrids.at[i, 'mg'] = mg                
                self.mv_microgrids.at[i, 'p_sgen_mg'] = (self.grid.sgen['p_mw'] * self.grid.sgen['scaling']).where(self.grid.sgen.bus.isin(mg), 0).sum()
                self.mv_microgrids.at[i, 'p_load_mg'] = (self.grid.load['p_mw'] * self.grid.load['scaling']).where(self.grid.load.bus.isin(mg), 0).sum()
        self.mv_microgrids['p_mw'] = self.mv_microgrids.groupby('mg')['weight'].transform('sum')
        
        if self.strg:
            if 'p_strg_a' not in self.mv_microgrids.columns:
                self.mv_microgrids['p_strg_b'] = 0
                self.mv_microgrids['p_strg_a'] = (self.mv_microgrids['p_sgen_mg'] - self.mv_microgrids['p_load_mg']).clip(lower=0)
                self.mv_microgrids['p_strg_a'] = np.where(self.mv_microgrids['weight'] != 0, self.mv_microgrids['p_strg_a'] * self.mv_microgrids['weight'] / self.mv_microgrids['p_mw'], 0)
                self.mv_microgrids['e_strg'] = 0
                self.mv_microgrids['e_strg_max'] = 0
                self.mv_microgrids['p_mv_mg'] = self.mv_microgrids[['p_sgen_mg', 'p_load_mg']].min(axis=1)
            else:
                self.mv_microgrids['p_strg_b'] = self.mv_microgrids['p_strg_a']
                self.mv_microgrids['p_strg_sum'] = (self.mv_microgrids.groupby('mg')['p_strg_b'].transform('sum', min_count=1))
                self.mv_microgrids['p_strg_sum'] = self.mv_microgrids['p_strg_sum'].fillna(self.mv_microgrids['p_strg_b'])
                
                self.mv_microgrids['p_mv_mg'] = np.where((self.mv_microgrids['p_load_mg'] - self.mv_microgrids['p_sgen_mg']) > self.mv_microgrids['p_strg_sum'], 
                                                         self.mv_microgrids['p_sgen_mg'] + self.mv_microgrids['p_strg_sum'], self.mv_microgrids['p_load_mg'])
                self.mv_microgrids['p_strg_sum'] = np.where((self.mv_microgrids['p_load_mg'] - self.mv_microgrids['p_sgen_mg']) >= self.mv_microgrids['p_strg_sum'],0,
                                                        self.mv_microgrids['p_strg_sum'] + self.mv_microgrids['p_sgen_mg'] - self.mv_microgrids['p_load_mg'])
                
                self.mv_microgrids['p_strg_a'] = np.where(self.mv_microgrids['weight'] != 0, self.mv_microgrids['p_strg_sum'] * self.mv_microgrids['weight'] / self.mv_microgrids['p_mw'], self.mv_microgrids['p_strg_b'])
                self.mv_microgrids['e_strg'] = np.where((self.mv_microgrids['p_strg_b'] - self.mv_microgrids['p_strg_a']) > 0,
                                                        self.mv_microgrids['e_strg'] + self.mv_microgrids['p_strg_b'] - self.mv_microgrids['p_strg_a'], 0) * 15/60
                self.mv_microgrids['e_strg_max'] = self.mv_microgrids[['e_strg_max', 'e_strg']].max(axis=1)
        else:
            self.mv_microgrids['p_mv_mg'] = self.mv_microgrids[['p_sgen_mg', 'p_load_mg']].min(axis=1)
        
        mv_mg_contribution = self.mv_microgrids.drop_duplicates(subset='mg', keep='first')['p_mv_mg'].sum()
   
        return mv_mg_contribution
    
    def simulate_scenario(self):
        self._reset()
        while True:
            # print((f"[TIME] Advanced to {self.current_time:.2f} h"))
            end_simulation = self._step()
            if end_simulation:
               return self._calculate_power_not_supplied()       


def create_grid():

        #collect data for components
        file_path_comp = 'components.xlsx'
        df_comp_transformers = pd.read_excel(file_path_comp, sheet_name='Transformers')
        df_comp_lines = pd.read_excel(file_path_comp, sheet_name='Lines')
        
        #collect data for the network
        file_path_grid = 'network.xlsx'
        df_buses = pd.read_excel(file_path_grid, sheet_name='Nodes')
        df_transformers = pd.read_excel(file_path_grid, sheet_name='Transformers')
        df_transformers = pd.merge(df_transformers, df_comp_transformers, on='Transf Type', how='left')
        df_lines = pd.read_excel(file_path_grid, sheet_name='Lines')
        df_lines = pd.merge(df_lines, df_comp_lines, on='Line Type', how='left')
        df_generators = pd.read_excel(file_path_grid, sheet_name='Static generators')
        df_loads = pd.read_excel(file_path_grid, sheet_name='Loads')
        
        #count_joints and terminals
        df_lines = df_lines.dropna(subset=['Line ID']).reset_index(drop=True)
        df_lines['N_terminals'] = 2
        df_lines['N_joints'] = df_lines['N_Branches'] - 1
        
        #create grid from parameters
        net = pp.create_empty_network(name='ATL_Rete_Urbana', f_hz=50.0, sn_mva=1e7)
        for b, row in df_buses.iterrows():
            pp.create_bus(net, index=b+1, name=row['Node ID'], vn_kv=row['Vn [kV]'], in_service=True)
            if b == 0:
                pp.create_ext_grid(net, name='External grid', bus=b+1)
        for t, row in df_transformers.iterrows():
            pp.create_transformer_from_parameters(net, name=row['Transf ID'], hv_bus=df_buses.index[df_buses['Node ID'] == row['HV Node']].tolist()[0]+1, lv_bus=df_buses.index[df_buses['Node ID'] == row['LV Node']].tolist()[0]+1, sn_mva=row['Sn [MVA]'], vn_hv_kv=row['Vrated1 [kV]'], vn_lv_kv=row['Vrated2 [kV]'], vk_percent=row['vcc [%]'], vkr_percent=row['pcc [%]'], pfe_kw=row['Sn [MVA]']*1e-3*0.2/100, i0_percent=1.0, in_service=True)    
        for l, row in df_lines.iterrows():
            if row['C [nF/km]'] < 50:
                pp.create_line_from_parameters(net, name=row['Line ID'], from_bus=df_buses.index[df_buses['Node ID'] == row['From']].tolist()[0]+1, to_bus=df_buses.index[df_buses['Node ID'] == row['To']].tolist()[0]+1, length_km=row['Length [km]'], r_ohm_per_km=row['R [Ohm/km]'], x_ohm_per_km=row['L [mH/km]']*2*math.pi*50*1e-3, c_nf_per_km=row['C [nF/km]'], r0_ohm_per_km=row['R0 [Ohm/km]'], x0_ohm_per_km=row['L0 [mH/km]']*2*math.pi*50*1e-3, c0_nf_per_km=row['C0 [nF/km]'], max_i_ka=row['Imax [A]']*1e-3, type='ol', in_service=True)
            else:
                pp.create_line_from_parameters(net, name=row['Line ID'], from_bus=df_buses.index[df_buses['Node ID'] == row['From']].tolist()[0]+1, to_bus=df_buses.index[df_buses['Node ID'] == row['To']].tolist()[0]+1, length_km=row['Length [km]'], r_ohm_per_km=row['R [Ohm/km]'], x_ohm_per_km=row['L [mH/km]']*2*math.pi*50*1e-3, c_nf_per_km=row['C [nF/km]'], r0_ohm_per_km=row['R0 [Ohm/km]'], x0_ohm_per_km=row['L0 [mH/km]']*2*math.pi*50*1e-3, c0_nf_per_km=row['C0 [nF/km]'], max_i_ka=row['Imax [A]']*1e-3, type='cs', in_service=True)
        for g, row in df_generators.iterrows():
            pp.create_sgen(net, name=row['Gen ID'], bus=df_buses.index[df_buses['Node ID'] == row['Node']].tolist()[0]+1, p_mw=row['P_oper [MW]'], q_mvar=row['Q_oper [MVar]'], in_service=True)
            if df_buses.index[df_buses['Node ID'] == row['Node']].tolist()[0]+1 not in net.ext_grid['bus'].values:
                pp.create_ext_grid(net, name='VS_'+row['Gen ID'], bus=df_buses.index[df_buses['Node ID'] == row['Node']].tolist()[0]+1, in_service=False)
        for c, row in df_loads.iterrows():
            pp.create_load(net, name=row['Load ID'], bus=df_buses.index[df_buses['Node ID'] == row['Node']].tolist()[0]+1, p_mw=row['Pn [MW]'], q_mvar=row['Qn [Mvar]'],  in_service=True)
        
        #calculate distances between the subtation and each node
        distances = pp.topology.calc_distance_to_bus(net,1)
        distances.index = distances.index - 1
        df_buses['Distance'] = distances
        
        return net, df_lines, df_buses, df_loads, df_generators
    
def upload_coefficients():
    
    #collect coefficients for load profiles
    file_path_profiles = 'profiles.xlsx'
    df_l_day_profile = pd.read_excel(file_path_profiles, sheet_name='Loads - daily profiles')
    df_l_week_profile = pd.read_excel(file_path_profiles, sheet_name='Loads - weekly profiles')
    df_l_month_profile = pd.read_excel(file_path_profiles, sheet_name='Loads - monthly profiles')
    df_l_year_profile = pd.read_excel(file_path_profiles, sheet_name='Loads - yearly profiles')
    
    #collect coefficients for generator profiles
    df_g_day_profile = pd.read_excel(file_path_profiles, sheet_name='Generators - daily profiles')
    df_g_week_profile = pd.read_excel(file_path_profiles, sheet_name='Generators - weekly profiles')
    df_g_month_profile = pd.read_excel(file_path_profiles, sheet_name='Generators - monthly profiles')
    df_g_year_profile = pd.read_excel(file_path_profiles, sheet_name='Generators - yearly profiles')
    
    return  df_l_day_profile, df_l_week_profile, df_l_month_profile, df_l_year_profile, df_g_day_profile, df_g_week_profile, df_g_month_profile, df_g_year_profile

    
N_iterations = 10000
epsilon = 0.01
R = pd.DataFrame(columns=['Iteration', 'Resilience index'])
b = pd.DataFrame(columns=['Iteration', 'beta'])
    
grid_base, lines_df, buses_df, loads_df, generators_df = create_grid()
l_day_profile_df, l_week_profile_df, l_month_profile_df, l_year_profile_df, g_day_profile_df, g_week_profile_df, g_month_profile_df, g_year_profile_df = upload_coefficients()

storage_df = grid_base.bus[['name']].copy()
storage_df['e_strg_max'] = 0

for i in range(N_iterations):
        
    np.random.seed(i)
        
    episode = OutageScenario(grid_base, lines_df, buses_df, loads_df, generators_df, l_day_profile_df, l_week_profile_df, 
                             l_month_profile_df, l_year_profile_df, g_day_profile_df, g_week_profile_df, g_month_profile_df, 
                             g_year_profile_df)
    resilience_trapezoid, power_profile, microgrids, results = episode.simulate_scenario()
    
    try:
        storage_df['e_strg_max'] = storage_df['e_strg_max'].where(storage_df['e_strg_max'] >= microgrids['e_strg_max'], microgrids['e_strg_max'])
    except:
        storage_df['e_strg_max'] = 0
    
    # plt.plot(resilience_trapezoid['Time'], resilience_trapezoid['Performance level'])
    # plt.ylim(bottom=0)
    # plt.show()
    
    ens = power_profile['P_not_supplied'].sum() * 15/60
    R.loc[len(R)] = [i + 1, ens]
    
    #check ending condition
    if  R['Resilience index'].sum() != 0 and i > 0:
        beta = np.sqrt(R['Resilience index'].var()) / (R['Resilience index'].mean() * np.sqrt(i + 1))
        b.loc[len(b)] = [i + 1, beta]
        print(beta)
        if beta <= epsilon:
            break

#calculate max required storage
storage_max = storage_df['e_strg_max'].sum()
    
#calculate mean resilience index
resilience_index = R['Resilience index'].mean()

results.at[0, 'Storage max'] = storage_max
results.at[0, 'Epsilon'] = epsilon
results.at[0, 'Iterations'] = i + 1
results.at[0, 'Resilience index'] = resilience_index

file_path = 'results.xlsx'
results.to_excel(file_path, sheet_name = 'Results',index=False)

