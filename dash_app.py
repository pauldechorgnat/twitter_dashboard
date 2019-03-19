import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go
import time
import string
import random
import pandas as pd

# defining colors for the app
app_colors = {
    'background': '#0C0F0A',
    'text': '#FFFFFF',
    'sentiment-plot': '#41EAD4',
    'volume-bar': '#FBFC74',
    'some-other-color': '#FF206E',
}

# defining colors for the size in the table
sentiment_colors = {-1: "#EE6055",
                    -0.5: "#FDE74C",
                    0: "#FFE6AC",
                    0.5: "#D0F2DF",
                    1: "#9CEC5B", }

# building the app
app = dash.Dash(__name__)

# building app layout
app.layout = html.Div(
    [
        html.Div(className='container-fluid',
                 children=[
                     html.H2('Live Twitter Sentiment',
                             style={'color': "#CECECE", 'text-align': 'center'}),
                     html.H5('Search:',
                             style={'color': app_colors['text']}),
                     dcc.Input(id='sentiment_term',
                               value='twitter',
                               type='text',
                               style={'color': app_colors['some-other-color']}),
                 ],
                 style={'width': '98%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000}),
        html.Div(className='row',
                 children=[
                     html.Div(id='related-sentiment',
                              children=html.Button('Loading related terms...',
                                                   id='related_term_button'),
                              className='col s12 m6 l6',
                              style={"word-wrap": "break-word"}),
                     html.Div(id='recent-trending',
                              className='col s12 m6 l6',
                              style={"word-wrap": "break-word"})
                 ]
                 ),
        html.Div(className='row',
                 children=[
                     html.Div(
                         dcc.Graph(id='live-graph',
                                   animate=False),
                         className='col s12 m6 l6'),
                     html.Div(dcc.Graph(id='historical-graph',
                                        animate=False),
                              className='col s12 m6 l6')
                 ]
                 ),

        html.Div(className='row',
                 children=[
                     html.Div(id="recent-tweets-table",
                              className='col s12 m6 l6'),
                     html.Div(children=dcc.Graph(id='sentiment-pie',
                                                 animate=False),
                              className='col s12 m6 l6'),
                 ]
                 ),
        dcc.Interval(
            id='graph-update',
            interval=1 * 1000
        ),
        dcc.Interval(
            id='historical-update',
            interval=60 * 1000
        ),

        dcc.Interval(
            id='related-update',
            interval=30 * 1000
        ),

        dcc.Interval(
            id='recent-table-update',
            interval=2 * 1000
        ),

        dcc.Interval(
            id='sentiment-pie-update',
            interval=60 * 1000
        ),
    ],
    style={
        'backgroundColor': app_colors['background'],
        'margin-top': '-30px',
        'height': '2000px',
    },
)


# updating pie chart
@app.callback(Output('sentiment-pie', 'figure'),
              [Input('sentiment-pie-update', 'n_intervals')])
def update_pie_chart(n):
    labels = ['Positive', 'Negative']
    pos, neg = np.random.randint(low=0, high=100, size=2)
    values = [pos, neg]
    colors = ['#007F25', '#800000']
    trace = go.Pie(labels=labels, values=values,
                   hoverinfo='label+percent', textinfo='value',
                   textfont=dict(size=20, color=app_colors['text']),
                   marker=dict(colors=colors,
                               line=dict(color=app_colors['background'], width=2)))

    return {"data": [trace], 'layout': go.Layout(
        title='Positive vs Negative sentiment" (longer-term)',
        font={'color': app_colors['text']},
        plot_bgcolor=app_colors['background'],
        paper_bgcolor=app_colors['background'],
        showlegend=True)}


# updating live graph
@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_live_graph(n):
    x = np.linspace(start=0, stop=100, num=100)
    y = np.random.randint(low=0, high=100, size=100)
    y2 = np.random.randint(low=0, high=100, size=100)
    data = go.Scatter(
        x=x,
        y=y,
        name='Sentiment',
        mode='lines',
        yaxis='y2',
        line=dict(color=(app_colors['sentiment-plot']),
                  width=4, )
    )

    data2 = go.Bar(
        x=x,
        y=y2,
        name='Volume',
        marker=dict(color=app_colors['volume-bar']),
    )

    return {'data': [data, data2], 'layout': go.Layout(xaxis=dict(range=[min(x), max(x)]),
                                                       yaxis=dict(range=[min(y2), max(y2 * 4)], title='Volume',
                                                                  side='right'),
                                                       yaxis2=dict(range=[min(y), max(y)], side='left',
                                                                   overlaying='y', title='sentiment'),
                                                       title='Live sentiment',
                                                       font={'color': app_colors['text']},
                                                       plot_bgcolor=app_colors['background'],
                                                       paper_bgcolor=app_colors['background'],
                                                       showlegend=False)}


# updating historical graph
@app.callback(Output('historical-graph', 'figure'),
              [Input('historical-update', 'n_intervals')])
def update_historical_graph(n):
    x = np.linspace(start=0, stop=100, num=100)
    y = np.random.randint(low=0, high=100, size=100)
    y2 = np.random.randint(low=0, high=100, size=100)
    data = go.Scatter(
        x=x,
        y=y,
        name='Sentiment',
        mode='lines',
        yaxis='y2',
        line=dict(color=(app_colors['sentiment-plot']),
                  width=4, )
    )

    data2 = go.Bar(
        x=x,
        y=y2,
        name='Volume',
        marker=dict(color=app_colors['volume-bar']),
    )
    return {'data': [data, data2],
            'layout': go.Layout(xaxis=dict(range=[min(x), max(x)]),  # add type='category to remove gaps'
                                yaxis=dict(range=[min(y2), max(y2 * 4)], title='Volume', side='right'),
                                yaxis2=dict(range=[min(y), max(y)], side='left', overlaying='y', title='sentiment'),
                                title='Longer-term sentiment',
                                font={'color': app_colors['text']},
                                plot_bgcolor=app_colors['background'],
                                paper_bgcolor=app_colors['background'],
                                showlegend=False)}


# defining a function to color text with respect to the sentiment
def quick_color(s, pos_neg_neut=0.1):
    # except return bg as app_colors['background']
    if s >= pos_neg_neut:
        # positive
        return "#002C0D"
    elif s <= -pos_neg_neut:
        # negative:
        return "#270000"

    else:
        return app_colors['background']


# defining a function to return a HTML table containing tweets
def generate_table(df, max_rows=10):
    return html.Table(className="responsive-table",
                      children=[
                          html.Thead(
                              html.Tr(
                                  children=[
                                      html.Th(col.title()) for col in df.columns.values],
                                  style={'color': app_colors['text']}
                              )
                          ),
                          html.Tbody(
                              [

                                  html.Tr(
                                      children=[
                                          html.Td(data) for data in d
                                      ], style={'color': app_colors['text'],
                                                'background-color': quick_color(d[2])}
                                  )
                                  for d in df.values.tolist()[:max_rows]])
                      ]
                      )


# updating the table containing the recent tweets
@app.callback(Output('recent-tweets-table', 'children'),
              [  # Input('sentiment_term', 'value'),
                  Input('recent-table-update', 'n_intervals')])
def update_recent_tweets_table(n):
    now_timestamp = int(time.time())
    table_size = 15
    dates = [str(pd.to_datetime(i))[:19] for i in
             np.random.randint(low=now_timestamp - 100000, high=now_timestamp, size=table_size)]
    tweets = [''.join(random.choices(string.printable, k=280)) for i in range(table_size)]
    sentiments = np.random.uniform(low=-1, high=1, size=15)
    df = pd.DataFrame(data={'date': dates, 'tweet': tweets, 'sentiment': sentiments})
    return generate_table(df, max_rows=10)


# defining a function to generate the size of the related terms
def generate_size(value, size_min, size_max, max_size_change=.4):
    size_change = round((((value - size_min) / size_max) * 2) - 1, 2)
    final_size = (size_change * max_size_change) + 1
    return final_size * 120


# updating the list of related terms
@app.callback(Output('related-sentiment', 'children'),
              [Input(component_id='sentiment_term', component_property='value')])
def update_related_sentiment(sentiment_term):
    number_of_related_terms = 10
    related_terms = [''.join(random.choices(string.printable, k=15)) for i in range(number_of_related_terms)]
    related_terms = dict(zip(related_terms, np.random.uniform(size=(number_of_related_terms, 2))))
    buttons = [html.Button('{}({})'.format(term, related_terms[term][1]), id='related_term_button', value=term,
                           className='btn', type='submit', style={'background-color': '#4CBFE1',
                                                                  'margin-right': '5px',
                                                                  'margin-top': '5px'}) for term in related_terms]
    sizes = [related_terms[term][1] for term in related_terms]
    size_min = min(sizes)
    size_max = max(sizes) - size_min

    buttons = [html.H5('Terms related to "{}": '.format(sentiment_term), style={'color': app_colors['text']})] + [
        html.Span(term, style={'color': sentiment_colors[round(related_terms[term][0] * 2) / 2],
                               'margin-right': '15px',
                               'margin-top': '15px',
                               'font-size': '{}%'.format(generate_size(related_terms[term][1], size_min, size_max))})
        for
        term in related_terms]

    return buttons


# updating the recently trading mentions
@app.callback(Output('recent-trending', 'children'),
              [Input(component_id='sentiment_term', component_property='value')])
def update_recent_trending(sentiment_term):
    number_of_related_terms = 10
    related_terms = [''.join(random.choices(string.printable, k=15)) for i in range(number_of_related_terms)]
    related_terms = dict(zip(related_terms, np.random.uniform(size=(number_of_related_terms, 2))))

    sizes = [related_terms[term][1] for term in related_terms]
    size_min = min(sizes)
    size_max = max(sizes) - size_min

    buttons = [html.H5('Recently Trending Terms: ', style={'color': app_colors['text']})] + \
              [html.Span(term,
                         style={'color': sentiment_colors[round(related_terms[term][0] * 2) / 2],
                                'margin-right': '15px',
                                'margin-top': '15px',
                                'font-size': '{}%'.format(generate_size(related_terms[term][1], size_min, size_max))})
               for
               term in related_terms]

    return buttons


# adding external css and js style sheets
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js',
               'https://pythonprogramming.net/static/socialsentiment/googleanalytics.js']
for js in external_js:
    app.scripts.append_script({'external_url': js})

if __name__ == '__main__':
    app.run_server(debug=True)
