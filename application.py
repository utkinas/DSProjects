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


def readModel():
    ds = pd.read_csv("data/diskStat3.csv").fillna(0)
    dfModels = pd.read_csv("data/modelList.csv")

    dsTest = ds[ds.capacity_bytes > 0]
    dsTest = pd.merge(left=dsTest, right=dfModels, how='left', left_on='model', right_on='model')


    simpleFeatures = ['capacity_bytes', 'smart_1_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw',
                      'smart_192_raw', 'smart_194_raw', 'smart_195_raw', 'smart_197_raw', 'smart_199_raw',
                      'smart_240_raw', 'smart_241_raw']

    # validate for another quarter
    Xtest = dsTest[simpleFeatures]
    Ytest = dsTest[['failure']]
    X1test = Xtest.fillna(0)





    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    predictions3 = loaded_model.predict(Xtest)
    probability = loaded_model.predict_proba(Xtest)
    #print('results after over sampling, logistic regressioin, no device filtering')
    #print(probability)
    #print(classification_report(Ytest, predictions3))
    #print(confusion_matrix(Ytest, predictions3))

    return round(100*round(probability[0][1],2))



dsTest = pd.read_csv("data/diskStat3.csv").fillna(0)

simpleFeatures = [ 'smart_1_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw',
                  'smart_12_raw', 'smart_192_raw', 'smart_194_raw', 'smart_195_raw', 'smart_197_raw', 'smart_199_raw',
                  'smart_240_raw', 'smart_241_raw']


df = pd.read_csv("data/diskStat3.csv")
df = df[simpleFeatures]


dfmeanUser = pd.DataFrame(df[simpleFeatures].mean()).rename(columns={0: 'userData'})
dfmeanUser = pd.DataFrame(list(dfmeanUser['userData'])).rename(columns={0: 'userData'})

dfmean = pd.read_csv("data/allmean.csv")

dfmean = pd.merge(left = dfmean,right=dfmeanUser, how = 'left', right_index=True,left_index=True)

names = list(dfmean['name'])
fail = list(dfmean['Fail'])
nonfail = list(dfmean['nonFail'])
userData = list(dfmean['userData'])
#df1 = df.head()


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.scripts.config.serve_locally = True
app.css.config.serve_locally = True


app.layout = html.Div(children=[
    html.H1(children='Health Dashboard'),

    html.Div(children=[
        'Hard drive statistics for today: ',
         generate_table(df)
             ]),


    html.H4(children='Probability of hard drive failure is '+ str(readModel())+'%'),

    html.H4(children='WARNING: Please back up your hard drive!'),


    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': names, 'y': fail, 'type': 'bar', 'name': 'failed'},
                {'x': names, 'y': nonfail, 'type': 'bar', 'name': 'non-failed'},
                {'x': names, 'y': userData, 'type': 'bar', 'name': 'userData'},
            ],
            'layout': {
                'title': 'Smart features distributions'
            }
        }
   )

])

application = app.server

if __name__ == '__main__':
    application.run(debug=True, port=8080)
