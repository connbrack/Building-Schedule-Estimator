import pandas as pd
import numpy as np
import warnings
import pwlf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


# ---------------- Supplementary Functions

def prep_data_pwlf(df_og, parameter, drop_zero=True):
    df = df_og.dropna(subset=['temperature', parameter]).copy()
    if drop_zero:
        df = df[df[parameter] != 0]
    
    date = df.index.to_numpy()
    x1 = df.temperature.to_numpy()
    x = pd.DataFrame({'date':date, 'x1':x1})
    y = df[parameter].to_numpy()
    return x, y


def generate_sch_dict(search_start, min_sch_len, weekend_sch=True):
    schs = []
    for work_day_start in range(search_start, 24-min_sch_len):
        for work_day_end in range(work_day_start + min_sch_len, 24):
            schs.append([float(work_day_start), float(work_day_end)])
    
    grid = {'sch': schs}
    
    if weekend_sch:
        weekend = schs.copy()
        weekend.append(None)

        grid = {
            'sch': schs,
            'weekend_sch': weekend
        }
    
    return grid

# ------------ sklearn style estimator

def separate_data_by_schedule(d_time, workday_sch, weekend_sch=None):
    """_summary_

    Args:
        d_time (datetime series): the datetime series of input variables
        workday_sch (list): A list of the weekday schedule in format [workday_start, workday_end] 
        weekend_sch (list, optional): A list of the weekend schedule in format [weekend_start, weekend_end]. Defaults to None.

    Returns:
        numpy array: numpy array with true
    """
    
    weekday_on = (d_time.dt.dayofweek < 5) & ((d_time.dt.hour > workday_sch[0]) & (d_time.dt.hour <= workday_sch[1]))
    
    if weekend_sch is None:
        return weekday_on.to_numpy()
    else:
        weekend_on= (d_time.dt.dayofweek >= 5) & ((d_time.dt.hour > weekend_sch[0]) & (d_time.dt.hour <= weekend_sch[1]))
        return (weekday_on | weekend_on).to_numpy()


class pwlf_sch(BaseEstimator, ClassifierMixin):
    def __init__(self, sch=None, weekend_sch=None, line_seg=2):
        '''
        '''
        self.sch = sch
        self.weekend_sch = weekend_sch
        self.line_seg = line_seg
        return None
        
    def fit(self, x, y):
        sch_split = separate_data_by_schedule(x.date, self.sch, weekend_sch=self.weekend_sch)
        x_on, y_on = x.x1[sch_split].to_numpy(), y[sch_split]
        x_off, y_off = x.x1[~sch_split].to_numpy(), y[~sch_split]
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.pwlf_on = pwlf.PiecewiseLinFit(x_on, y_on)
            self.pwlf_on.fit(self.line_seg)
            self.pwlf_off = pwlf.PiecewiseLinFit(x_off, y_off)
            self.pwlf_off.fit(self.line_seg)    
               
        return self
    
    def predict(self, x, ignore_sch=False):
        # Split Schedules
        if ignore_sch:
            sch_split = np.full(len(x.date), True, dtype=bool) # Create fully True Data Frame
        else:
            sch_split = separate_data_by_schedule(x.date, self.sch, weekend_sch=self.weekend_sch)               
        
        # Apply Model to appropriate columns
        y_pred = x.x1.copy()     
        y_pred[sch_split] = self.pwlf_on.predict(y_pred[sch_split])
        y_pred[~sch_split] = self.pwlf_off.predict(y_pred[~sch_split])
        y_pred = y_pred.to_numpy()
        
        return y_pred
    
    def RMSE(self, x, y, normalize=False):
        if normalize:
            return 1 / self.score(x,y) / y.mean()
        else:
            return 1 / self.score(x,y)
    
    def score(self, x, y):
        """Score is 1/RMSE to support best_estimator_ function in grid search"""
        y_pred = self.predict(x)
        return 1 / mean_squared_error(y, y_pred, squared=False)
    
    def est_savings(self, x):
        standard = self.predict(x, ignore_sch=False).sum()
        no_sch = self.predict(x, ignore_sch=True).sum()
        diff = no_sch - standard
        aver = (no_sch + standard) / 2
        return diff / aver
    
    
# ------------ Modified grid search for schedule finding
    
class sch_search():
    """Modified grid search to so that weekend and weekday schedules are evaluated individually"""
    def __init__(self, pwlf_sch_estimator, param_grid, n_jobs=None, verbose=None):
        self.pwlf_sch_estimator = pwlf_sch_estimator
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.verbose = verbose
        
    def fit(self, x, y):
        if 'weekend_sch' not in self.param_grid:
            weekday_search = GridSearchCV(
                estimator=self.pwlf_sch_estimator, 
                param_grid=self.param_grid,
                n_jobs=self.n_jobs,
                cv=[(slice(None), slice(None))],
                verbose=self.verbose
            ).fit(x, y)
            return weekday_search.best_estimator_
        else:
            x_weekday = x[(x.date.dt.dayofweek < 5)]
            y_weekday = y[(x.date.dt.dayofweek < 5)]
            gid_weekday = {'sch': self.param_grid['sch']}
            if self.verbose != 0:
                print('Searching for best weekday schedule')
            weekday_search = GridSearchCV(
                estimator=self.pwlf_sch_estimator, 
                param_grid=gid_weekday,
                n_jobs=self.n_jobs,
                cv=[(slice(None), slice(None))],
                verbose=self.verbose
            ).fit(x_weekday, y_weekday)
            
            grid_weekday = {'sch': [weekday_search.best_estimator_.sch], 'weekend_sch': self.param_grid['weekend_sch']}
            if self.verbose != 0:
                print('Searching for best weekend schedule')
            full_search = GridSearchCV(
                estimator=self.pwlf_sch_estimator, 
                param_grid=grid_weekday,
                n_jobs=self.n_jobs,
                cv=[(slice(None), slice(None))],
                verbose=self.verbose
            ).fit(x, y)
            return full_search.best_estimator_