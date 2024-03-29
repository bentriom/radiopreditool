
import dash
from dash import dcc, dash_table, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
DATASET_DIR = "../database/fccss_01122022/"
# Data
# df_events = pd.read_csv(DATASET_DIR + "events_fccss_igr_curie_011021.csv", low_memory = False)
df_events = pd.read_csv(DATASET_DIR + "base_fccss_igr_curie_011222_extended.csv.gz", low_memory = False)
df_labels = pd.read_csv(DATASET_DIR + "dictionnary_variables_fccss_igr_curie_01102021.csv", low_memory = False)
# df_iccc = pd.read_csv(DATASET_DIR + "iccc.csv", low_memory = False)

# About column events
benign_events = [f"K2_loc{nbr_loc}" for nbr_loc in range(45,58)]
non_cancer_events = ["Pathologie_auditive", "Pathologie_cardiaque", "Pathologie_cardiaque_3", \
                     "Pathologie_renale_chronique", "pathologie_cataracte", "pathologie_cerebrovasculaire", \
                     "pathologie_chir_cataracte", "pathologie_diabete"]
non_benign_events = [f"K2_loc{nbr_loc}" for nbr_loc in range(1,45)]
sorted_columns = non_benign_events + benign_events + non_cancer_events

# Op. dataframes
# df_merge = pd.merge(df_events, df_iccc, how = "inner", on = ["ctr", "numcent"])
df_events = df_events[non_cancer_events + non_benign_events + ["iccc", "iccc_lab"]]
df_labels = df_labels.set_index("Variable Name")

nbr_patients = df_events.shape[0]
nbr_patients_events = df_events.sum(numeric_only = True)
proportion_events = 100 * df_events.sum(numeric_only = True) / nbr_patients
max_prop = max(proportion_events)
list_iccc_lab = df_events["iccc_lab"].unique()
discrete_colors_pie = (px.colors.qualitative.Set1 + px.colors.qualitative.Dark2 +
                       px.colors.qualitative.Bold + px.colors.qualitative.Antique +
                       px.colors.qualitative.Safe)[0:len(list_iccc_lab)]
map_labels_colors_pie = dict(zip(list_iccc_lab, discrete_colors_pie))
map_labels_colors_pie[np.nan] = "#000000"

app.layout = html.Div(children=[
    html.H1("Data visualisation for the outcomes of FCCSS"),
    #html.P(children=["List of available columns:", 
    #    dcc.Dropdown(id='list_cols', style={'width': '300px'}, 
    #    options=[{'label': col, 'value': col} for col in sorted_columns])]),
    html.H2("Type of events"),
    dcc.RadioItems(
    id="event_type",
    options=[
        {'label': 'Non-benign tumors', 'value': 'non_benign'},
        {'label': 'Non-cancer pathology', 'value': 'non_cancer'}
    ],
    value='non_benign'),
    html.H2("Proportion of occured events with a threshold filter"),
    html.Div(children=[
        dcc.Slider(
            id='threshold_slider',
            min=0.0,
            max=max_prop,
            marks={val: f'{val:.2f}' for val in [0, 0.5, 1, 2, 5, max_prop]},
            step=0.05,
            value=1)],
        style={'width':'90%'}),
    html.Div(id='out_text_threshold'),

    html.H2("Proportion of diagnosed first cancer per outcome"),
    html.Div(id='first_cancer_table'),
    dcc.Dropdown(id='outcome',
                 options=[
                 {'label': f'{df_labels.loc[loc]["Variable Label"]} ({loc})', 'value': loc} for loc in sorted_columns
                 ] + [{'label': 'All', 'value': 'All'}],
                 value='K2_loc40'),
    html.Div(id='first_cancer_analysis')
    #dcc.Graph(id='first_cancer_pie')
])

def get_list_events(event_type):
    if event_type == "non_benign":
        list_events = non_benign_events
    elif event_type == "non_cancer":
        list_events = non_cancer_events
    else:
        raise ValueError("Type of event non handled")
    return list_events

@app.callback(
    Output(component_id='outcome', component_property='options'),
    Output(component_id='outcome', component_property='value'),
    Input(component_id='event_type', component_property='value'),
    Input(component_id='threshold_slider', component_property='value'),
)
def update_event_type(event_type, threshold_slider):
    list_events = get_list_events(event_type)
    filter_proportion_events = proportion_events.loc[list_events]
    above_threshold_events = filter_proportion_events[filter_proportion_events >= threshold_slider]
    sorted_events = above_threshold_events.sort_values(ascending=False)
    return ([{'label': f'{loc} - {"" if (type(df_labels.loc[loc, "Variable Label"]) != str) else df_labels.loc[loc, "Variable Label"]}',
              'value': loc} \
             for loc in sorted_events.index], sorted_events.index[0])

@app.callback(
    Output(component_id='out_text_threshold', component_property='children'),
    Input(component_id='event_type', component_property='value'),
    Input(component_id='threshold_slider', component_property='value'),
)
def update_out_text_slider(event_type, threshold_slider):
    list_events = get_list_events(event_type)
    filter_proportion_events = proportion_events.loc[list_events]
    above_threshold_events = filter_proportion_events[filter_proportion_events >= threshold_slider]
    sorted_events = above_threshold_events.sort_values(ascending=False)
    dict_table = {'Id': sorted_events.index,
                  'Label': df_labels.loc[sorted_events.index]["Variable Label"],
                  'Number of patients': nbr_patients_events.loc[sorted_events.index],
                  'Proportion (in %)': sorted_events.round(decimals=3)}
    df_table = pd.DataFrame(dict_table)
    '''
    list_li_events = []
    for idx in idx_sorted_events:
        id_event = above_threshold_events.index[idx]
        label_event = df_labels.loc[id_event,'Variable Label']
        list_li_events.append(html.Li(f'Id: {id_event}, Label: {label_event}  ({above_threshold_events[idx]:.2f})%'))
    return html.P(children=[f'Threshold: {threshold_slider}',
                            html.Ul(children=list_li_events)])
    '''
    level_patient = threshold_slider/100*nbr_patients
    level_patient = int(level_patient) if level_patient == int(level_patient) else int(level_patient)+1
    return html.Div(children=[html.Br(),
                              html.P(f'Number of patients in the cohort: {nbr_patients}.'),
                              html.P(f'Threshold: {threshold_slider}% (at least {level_patient} patients per event)'),
                              html.P(f'Number of events: {len(sorted_events)}'),
                              dash_table.DataTable(id='table_events',
                                                  columns=[{"name": col, "id": col} for col in df_table.columns],
                                                  data=df_table.to_dict('records'))])

@app.callback(
    Output(component_id='first_cancer_table', component_property='children'),
    Input(component_id='event_type', component_property='value'),
    Input(component_id='threshold_slider', component_property='value'),
)
def update_first_cancer_table(event_type, threshold_slider):
    list_events = get_list_events(event_type)
    filter_proportion_events = proportion_events.loc[list_events]
    above_threshold_events = filter_proportion_events[filter_proportion_events >= threshold_slider]
    sorted_events = above_threshold_events.sort_values(ascending=False)
    list_outcomes = sorted_events.index
    df_table = pd.DataFrame(index=list_iccc_lab, columns=list_outcomes, dtype='int').fillna(0)
    for outcome in list_outcomes:
        counts = df_events[df_events[outcome] == 1].groupby('iccc_lab').size()
        df_table.loc[counts.index, outcome] = counts.values
    total_patients = df_table.sum(numeric_only = True)
    df_table.loc["All", total_patients.index] = total_patients.values
    df_table = df_table.reset_index()

    return html.Div(children=[dash_table.DataTable(id='table_events',
            columns=[{"name": col, "id": col} for col in df_table.columns],
            data=df_table.to_dict('records'), style_cell={'textAlign': 'left'},), html.Br()])

"""
Function that updates the repartition of the first cancer among the population that experencied a specific outcome.
Requires:
    - (str) outcome: the second event
Guarantees:
    - (html.Div) HTML component with two psie chart of the first cancer (subpopulation of the outcome, all the FCCSS)
"""
@app.callback(
    Output(component_id='first_cancer_analysis', component_property='children'),
    Input(component_id='outcome', component_property='value')
)
def update_pie_first_cancer(outcome):
    # First cancer pie chart for the outcome subpopulation
    query_df = df_events[df_events[outcome] == 1]
    query_df = query_df.groupby('iccc_lab').size()
    labels_colors = [map_labels_colors_pie[label] for label in query_df.index]
    marker_pie = dict(colors=labels_colors)
    data_fig_outcome = [go.Pie(labels=query_df.index.values, values=query_df.values, marker=marker_pie)]
    fig_outcome = go.Figure(data=data_fig_outcome)
    fig_outcome.update_traces(textinfo='value', hoverinfo='label+percent', textposition='inside')
    fig_outcome.update_layout(title_text=f'First cancer repartition for {outcome} outcome')
    # First cancer pie chart for all FCCSS survivors
    query_df = df_events.groupby('iccc_lab').size()
    labels_colors = [map_labels_colors_pie[label] for label in query_df.index]
    marker_pie = dict(colors=labels_colors)
    data_fig_all = [go.Pie(labels=query_df.index.values, values=query_df.values, marker=marker_pie)]
    fig_all = go.Figure(data=data_fig_all)
    fig_all.update_traces(textinfo='value', hoverinfo='label+percent', textposition='inside')
    fig_all.update_layout(title_text=f'First cancer repartition all FCCSS survirors')

    str_description = f'{int(nbr_patients_events.loc[outcome])} patients' +\
                      f' ({proportion_events.loc[outcome]:.2f}% of the cohort)'
    return html.Div(children=[html.P(str_description),
                              dcc.Graph(id='first_cancer_pie_outcome', figure=fig_outcome),
                              dcc.Graph(id='first_cancer_pie_all', figure=fig_all)])

if __name__ == '__main__':
    app.run_server(debug=True)

