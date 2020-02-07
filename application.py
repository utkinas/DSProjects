import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
#import hddHealth
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


def readModel(testFileName):

    #read user test file
    ds = pd.read_csv("~/data/diskStat_dc.csv").fillna(0)

    ds = ds[ds.capacity_bytes > 0]

    simpleFeatures = ['capacity_bytes', 'smart_1_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw',
                      'smart_188_raw', 'smart_189_raw', 'smart_190_raw', 'smart_192_raw', 'smart_194_raw',
                      'smart_195_raw', 'smart_197_raw', 'smart_199_raw',
                      'smart_200_raw', 'smart_240_raw', 'smart_241_raw']

    # validate for user data
    Xtest = ds[simpleFeatures]
    Ytest = ds[['failure']]
    X1test = Xtest.fillna(0)
    info = ds[['date', 'model', 'serial_number', 'capacity_bytes']]

    filename_RF = 'finalized_model_RF.sav'
    filename_LR = 'finalized_model.sav'

    loaded_model = pickle.load(open(filename_RF, 'rb'))

    predictions = loaded_model.predict(X1test)
    probability = loaded_model.predict_proba(X1test)

    user_prob = pd.DataFrame(probability).rename(columns={1: 'probability'})
    result = pd.merge(Ytest, user_prob[['probability']], right_index=True, left_index=True)


    predRF = pd.DataFrame(predictions).rename(columns={0: 'predictions'})
    result = pd.merge(result, predRF, right_index=True, left_index=True)

    result = pd.merge(result, info[['model', 'serial_number', 'capacity_bytes']], right_index=True,
                      left_index=True).sort_values(by='probability', ascending=False)
    # result.to_csv("~/data/dashboardResults.csv")

    result['probability']=[round(x,2) for x in result['probability']]
    serialNumbersOfFailing = list(result[result.predictions == 1]['serial_number'])


    disksLikelyToFail = result[result.predictions == 1].head()



    #get user data for the device with highest probability of failure
    dfUser = ds[ds.serial_number == serialNumbersOfFailing[0]]

    topFeatures = pd.read_csv("~/data/top5Features.csv").head()[['name', 'fullName']]
    topFeaturesArr = topFeatures[['name']]
    topFeaturesList = list(topFeaturesArr['name'])
    topFeatures['name_y'] = topFeatures['name']

    dfmean = pd.read_csv("data/allmean.csv")

    dfmeanTopFeatures = dfmean[dfmean.name.isin(topFeaturesList)]
    dfmeanUser = pd.DataFrame(dfUser[simpleFeatures].mean()).rename(columns={0: 'userData'})
    dfmeanUser['name_x'] = dfmeanUser.index
    dfmeanUserTopFeatures = dfmeanUser[dfmeanUser.name_x.isin(topFeaturesList)]

    dfmeanTopFeatures = pd.merge(left=dfmeanTopFeatures, right=dfmeanUserTopFeatures, how='left', right_on='name_x',
                                 left_on='name')

    dfmeanTopFeatures = pd.merge(left=dfmeanTopFeatures, right=topFeatures, how='left', right_on='name_y',
                                 left_on='name_x')



    # graph to display
    dfmeanTopFeatures = dfmeanTopFeatures[['fullName', 'Fail', 'nonFail', 'userData']]






    return dfmeanTopFeatures, serialNumbersOfFailing, disksLikelyToFail[['serial_number','model']]



dfmeanTopFeatures, serialNumbersOfFailing, disksLikelyToFail = readModel("data/diskStat_dc.csv")

names = list(dfmeanTopFeatures['fullName'])
fail = list(dfmeanTopFeatures['Fail'])
nonfail = list(dfmeanTopFeatures['nonFail'])
userData = list(dfmeanTopFeatures['userData'])
#df1 = df.head()


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#BCE4FB',
    'text': '#000000'
}

 # #

#app.scripts.config.serve_locally = True
#app.css.config.serve_locally = True


app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(children='Health Dashboard',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
            ),

html.H2(children='WARNING: Please back up or replace these hard drives: ',
        style={
            'textAlign': 'center',

            'color': '#FF0000'
            }
        ),

    html.Div(  children=[

         generate_table(disksLikelyToFail)
             ],
        style={
            'textAlign': 'center',

            'color': colors['text']
            }


    ),


    #html.H4(children='Probability of hard drive failure is '+ str(readModel())+'%'),

    html.H4(children='S.M.A.R.T. features distribution',
                style={
                'textAlign': 'center',
                'color': colors['text']
            })
    ,


    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': names, 'y': fail, 'type': 'bar', 'name': 'failed'},
                {'x': names, 'y': nonfail, 'type': 'bar', 'name': 'non-failed'},
                {'x': names, 'y': userData, 'type': 'bar', 'name': 'userData'},
            ],
            'layout': {
                #'title': 'S.M.A.R.T. features distribution'
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }

            }
        }
   )

])


# local host http://127.0.0.1:8050/
application = app.server

if __name__ == '__main__':
    application.run(debug=True, port=8080)

#if __name__ == '__main__':
    #app.run_server(debug=True)