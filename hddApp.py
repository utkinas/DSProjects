import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


df = pd.read_csv("")
df1 = df.head()


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Health Dashboard'),

    html.Div(children=[
        'Hard drive statistics for today: ',
         generate_table(df)
             ]),


    html.H1(children='Probability of hard drive failure is 50%'),

    html.H1(children='WARNING: Please back up your hard drive!')



   # dcc.Graph(
   #     id='example-graph',
   #     figure={
   #         'data': [
   #             {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
   #             {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
   #         ],
   #         'layout': {
   #             'title': 'Dash Data Visualization'
   #         }
   #     }
#)

])

if __name__ == '__main__':
    app.run_server(debug=True)