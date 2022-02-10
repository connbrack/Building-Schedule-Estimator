import pandas as pd
import pathlib
import numpy as np
import pickle as plk
import pwlf
import os


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = np.mean(ys_orig)
    squared_error_res = np.sum((ys_orig - ys_line) ** 2)
    squared_error_tot = np.sum((ys_orig - y_mean_line) ** 2)
    return 1 - (squared_error_res / squared_error_tot)

def LinModel(Data, sch, parameter, WeekendSplit=True):
    # clean data for
    linData = Data.copy()
    linData = linData.dropna(subset=['temperature', parameter])
    linData = linData[linData[parameter] != 0]

    linDataOnSchedule, linDataOffSchedule = seperate_data_by_Schedule(linData, sch[0], sch[1], WeekendSplit)

    X_On = linDataOnSchedule['temperature'].to_numpy()
    Y_On = linDataOnSchedule[parameter].to_numpy()
    X_Off = linDataOffSchedule['temperature'].to_numpy()
    Y_Off = linDataOffSchedule[parameter].to_numpy()

    ## Find Change Point Models
    pwlf_On = pwlf.PiecewiseLinFit(linDataOnSchedule['temperature'].to_numpy(), linDataOnSchedule[parameter].to_numpy())
    cpts_On = pwlf_On.fit(3)
    pwlf_Off = pwlf.PiecewiseLinFit(linDataOffSchedule['temperature'].to_numpy(),
                                    linDataOffSchedule[parameter].to_numpy())
    cpts_Off = pwlf_Off.fit(3)

    Y_On_Fit = pwlf_On.predict(X_On)
    Y_Off_Fit = pwlf_Off.predict(X_Off)

    r2_tot = coefficient_of_determination(np.concatenate([Y_On, Y_Off]), np.concatenate([Y_On_Fit, Y_Off_Fit]))

    return pwlf_On, pwlf_Off, r2_tot

    
    
    
def sch_tester(data, sch, parameter, weekend_split):
    pwlf_On, pwlf_Off, r2_tot = LinModel(data, sch, parameter, weekend_split)
    results.append([sch[0],sch[1],r2_tot])