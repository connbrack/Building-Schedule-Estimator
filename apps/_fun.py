import pandas as pd
import numpy as np
import pwlf
import pathlib
import pickle as plk
import plotly.express as px
import plotly.graph_objects as go


def seperate_data_by_Schedule(Data, workDayStart, workDayEnd, WeekendSplit):
    if WeekendSplit:
        DataWeekend = Data[Data.index.weekday > 4]
        DataWeekday = Data[Data.index.weekday < 5]
        DataEvening = DataWeekday[(DataWeekday.index.hour <= workDayStart) | (DataWeekday.index.hour > workDayEnd)]
        DataOffSchedule = pd.concat([DataWeekend, DataEvening])
        DataOnSchedule = DataWeekday[(DataWeekday.index.hour > workDayStart) & (DataWeekday.index.hour <= workDayEnd)]

    else:
        DataOffSchedule = Data[(Data.index.hour <= workDayStart) | (Data.index.hour > workDayEnd)]
        DataOnSchedule = Data[(Data.index.hour > workDayStart) & (Data.index.hour <= workDayEnd)]

    return DataOnSchedule, DataOffSchedule


def MonthlyAggrigate(Data, resampeStep, columName, conversion):
    Argg = Data[['Timestamp', columName]]  # Get only electricity
    Argg[columName] = Argg[columName] * conversion  # Convert from MJ per hour to kWh
    Argg = Argg.resample(resampeStep, on='Timestamp', closed='left').sum()  # Filter by every 3 months
    return Argg


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = np.mean(ys_orig)
    squared_error_res = np.sum((ys_orig - ys_line) ** 2)
    squared_error_tot = np.sum((ys_orig - y_mean_line) ** 2)
    return 1 - (squared_error_res / squared_error_tot)


# --------------------------------- Page 2: Determine Schedule
def schedule_seperate_graph(building, year, data, sch, parameter, color_on, color_off, fit_line, color_line_on,
                            color_line_off, weekend_included=True):
    """
    :type color_line_on:
    :param color_line_on:
    :param data: Dataframe with combined energy data and
    :param sch: Start and end of occupied period as dictionary [start, end]
    :param parameter: steam/chilledWater/electrical
    :param color_on:
    :param color_off:
    :return: Figure with coloured
    """

    DataOnSchedule, DataOffSchedule = seperate_data_by_Schedule(data, sch[0], sch[1], weekend_included)

    fig = go.Figure()
    fig.add_scatter(name='Off Schedule', x=DataOffSchedule.temperature, y=DataOffSchedule[parameter],
                    customdata=DataOffSchedule.index, mode='markers', marker=dict(color=color_off, opacity=0.7))
    fig.add_scatter(name='On Schedule', x=DataOnSchedule.temperature, y=DataOnSchedule[parameter],
                    customdata=DataOnSchedule.index, mode='markers', marker=dict(color=color_on, opacity=0.7))

    fit_line = sort_out_fit_line(building, year, sch, parameter)
    if fit_line != []:
        fig.add_scatter(x=fit_line[0], y=fit_line[1], marker=dict(color=color_line_on), showlegend=False)
        fig.add_scatter(x=fit_line[2], y=fit_line[3], marker=dict(color=color_line_off), showlegend=False)

    fig.update_layout(template="none", xaxis_title="Outdoor Temperature", yaxis_title="Energy (GJ/hr)")
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    fig.update_traces(hovertemplate='Energy: %{y}<br>Date: %{customdata}')
    return fig


def sort_out_fit_line(building, year, sch, parameter):
    ## ------------------ Fit Line
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("../saved").resolve()
    with open(DATA_PATH.joinpath(f'model.pkl'), 'rb') as f:
        model = plk.load(f)
    model_building, model_year, steam_sch, steam_model_on, steam_model_off, _, water_sch, water_model_on, water_model_off, _, elec_sch, elec_model_on, elec_model_off, _ = model

    fit_line = []
    if model_building == building and model_year == year:
        if parameter == 'steam':
            if steam_sch == sch:
                fit_line = [steam_model_on.fit_breaks, steam_model_on.predict(steam_model_on.fit_breaks),
                            steam_model_off.fit_breaks, steam_model_off.predict(steam_model_off.fit_breaks)]
        elif parameter == 'chilledWater':
            if water_sch == sch:
                fit_line = [water_model_on.fit_breaks, water_model_on.predict(water_model_on.fit_breaks),
                            water_model_off.fit_breaks, water_model_off.predict(water_model_off.fit_breaks)]
        else:
            if elec_sch == sch:
                fit_line = [elec_model_on.fit_breaks, elec_model_on.predict(elec_model_on.fit_breaks),
                            elec_model_off.fit_breaks, elec_model_off.predict(elec_model_off.fit_breaks)]
    return fit_line


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


# --------------------------------- Page 3: Determine Schedule

def Energy_bar_graph(data, sch, weekendSplit, parameter):
    data = pd.read_json(data, typ='series', orient='split')
    if weekendSplit == [False]:
        weekendSplit = False
    else:
        weekendSplit = True

    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("../saved").resolve()

    with open(DATA_PATH.joinpath(f'model.pkl'), 'rb') as f:
        model = plk.load(f)
    _, _, steam_sch, steam_model_on, steam_model_off, _, water_sch, water_model_on, water_model_off, _, elec_sch, elec_model_on, elec_model_off, _ = model

    if parameter == 'steam':
        sch_model, model_on, model_off = steam_sch, steam_model_on, steam_model_off
    elif parameter == 'chilledWater':
        sch_model, model_on, model_off = water_sch, water_model_on, water_model_off
    else:
        sch_model, model_on, model_off = elec_sch, elec_model_on, elec_model_off

    no_sch = model_on.predict(data).sum()

    data_on_sch, data_off_sch = seperate_data_by_Schedule(data, sch_model[0], sch_model[1], True)
    prev_sch = model_on.predict(data_on_sch).sum() + model_off.predict(data_off_sch).sum()

    data_on_sch, data_off_sch = seperate_data_by_Schedule(data, sch[0], sch[1], weekendSplit)
    proposed_sch = model_on.predict(data_on_sch).sum() + model_off.predict(data_off_sch).sum()

    d = {'No Setback Schedule': [no_sch], 'Modeled Setback Schedule': [prev_sch], 'Proposed Setback Schedule': [proposed_sch]}
    result_table = pd.DataFrame(data=d)
    fig = px.bar(result_table.T)
    fig.update_layout(template="simple_white", yaxis_title="Energy (GJ)", showlegend=False, xaxis_title=None)

    savings = [prev_sch, proposed_sch]

    return fig, savings

