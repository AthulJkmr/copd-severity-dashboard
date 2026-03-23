from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier

# --- Load & prepare data ---
df = pd.read_csv('data/dataset.csv')
y = df['COPDSEVERITY']
X = df.drop(columns=['COPDSEVERITY', 'copd', 'ID'], errors='ignore')
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# --- Preprocessing pipelines ---
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipe, numeric_cols),
    ('cat', categorical_pipe, categorical_cols),
])

# --- Train/test split & encode ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# --- XGBoost model ---
xgb_pipe = Pipeline([
    ('prep', preprocessor),
    ('classifier', XGBClassifier(eval_metric='mlogloss', random_state=42, 
                               n_estimators=100, learning_rate=0.1, max_depth=5))
])
xgb_pipe.fit(X_train, y_train_enc)
xgb_acc = accuracy_score(y_test_enc, xgb_pipe.predict(X_test))

# --- Random Forest baseline & accuracy ---
rf_pipe = Pipeline([
    ('prep', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_pipe.fit(X_train, y_train_enc)
rf_acc = accuracy_score(y_test_enc, rf_pipe.predict(X_test))

# --- For WHO data visualization ---
# Load WHO data (if available)
try:
    who_df = pd.read_csv('data/WHO.csv')
    has_who_data = True
except:
    has_who_data = False

def filter_df(df_, sev=None, gen=None, age_rng=None):
    d = df_.copy()
    if sev is not None: d = d[d['COPDSEVERITY'].isin(sev)]
    if gen is not None: d = d[d['gender'].isin(gen)]
    if age_rng is not None: d = d[d['AGE'].between(age_rng[0], age_rng[1])]
    return d

# --- Dash App Setup ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# --- Layout with Tabs ---
app.layout = dbc.Container(fluid=True, style={'padding':'2rem'}, children=[
    # Dashboard Heading
    html.H1("COPD Global Impact & Patient Insights Dashboard", className="text-center mb-4"),
    html.P("From individual patient assessment to global health impact", 
           className="text-center text-muted mb-5"),

    # Tabs Navigation
    dbc.Tabs([
        # Overview Tab
        dbc.Tab(label="Overview", tab_id="overview", children=[
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H4("COPD at a Glance"),
                    html.P("Chronic Obstructive Pulmonary Disease (COPD) is a progressive lung disease that causes breathing difficulty. It affects over 250 million people worldwide and is the third leading cause of death globally."),
                    
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H5("Patient Assessment"),
                                html.P("COPD severity is measured through lung function tests (FEV1), exercise capacity (6-minute walk), and symptom assessments."),
                            ])
                        ], color="info", outline=True), md=6),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H5("AI-Powered Care"),
                                html.P("Machine learning models can predict COPD progression and help personalize treatment plans based on patient data."),
                            ])
                        ], color="success", outline=True), md=6),
                    ], className="mt-3 mb-3"),
                    
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H5("Risk Factors"),
                                html.P("Smoking history, air pollution exposure, occupational hazards, and genetic factors all contribute to COPD risk."),
                            ])
                        ], color="warning", outline=True), md=6),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H5("Global Burden"),
                                html.P("COPD prevalence and mortality vary significantly by region, age group, and sex across the globe."),
                            ])
                        ], color="danger", outline=True), md=6),
                    ]),
                ])), md=6),
                
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H4("Dashboard Navigation Guide"),
                    html.P("This dashboard presents a comprehensive story of COPD from clinical assessment to global impact."),
                    
                    html.Div([
                        html.Div([
                            html.Span("1", className="bg-primary text-white rounded-circle p-2 mr-2"),
                            html.Span("Patient Level", className="font-weight-bold"),
                            html.P("Understand how COPD severity relates to functional capacity and AI model performance.", 
                                  className="ml-4 text-muted small")
                        ], className="mb-3"),
                        
                        html.Div([
                            html.Span("2", className="bg-primary text-white rounded-circle p-2 mr-2"),
                            html.Span("Demographics", className="font-weight-bold"),
                            html.P("Explore how COPD affects different age groups and sexes.", 
                                  className="ml-4 text-muted small")
                        ], className="mb-3"),
                        
                        html.Div([
                            html.Span("3", className="bg-primary text-white rounded-circle p-2 mr-2"),
                            html.Span("Global Impact", className="font-weight-bold"),
                            html.P("View worldwide trends, regional disparities, and country-specific mortality rates.", 
                                  className="ml-4 text-muted small")
                        ], className="mb-3"),
                        
                        html.Div([
                            html.Span("4", className="bg-primary text-white rounded-circle p-2 mr-2"),
                            html.Span("Advanced Analysis", className="font-weight-bold"),
                            html.P("Discover how COPD mortality patterns have evolved over time across demographic segments.", 
                                  className="ml-4 text-muted small")
                        ]),
                    ]),
                ])), md=6),
            ]),
        ], className="p-4"),

        # Patient Level Tab
        dbc.Tab(label="Patient Level", tab_id="patient", children=[
            dbc.Row([
                # Model Performance Cards
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("XGBoost CV Best Accuracy"),
                    html.P(f"{xgb_acc:.2%}")
                ]), color="success", inverse=True), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Random Forest Test Accuracy"),
                    html.P(f"{rf_acc:.2%}")
                ]), color="info", inverse=True), md=6),
            ], className="mb-4"),
            
            # FEV1 vs. 6-Min Walk
            html.H4("FEV1 vs. 6-Min Walk"),
            dbc.Card([
                dbc.CardHeader(dbc.Row([
                    dbc.Col(dcc.Dropdown(
                        id='sc_sev', multi=True,
                        options=[{'label': s, 'value': s} for s in sorted(df['COPDSEVERITY'].unique())],
                        value=sorted(df['COPDSEVERITY'].unique()),
                        placeholder="Severity"
                    ), width=4),
                    dbc.Col(dcc.Dropdown(
                        id='sc_gen', multi=True,
                        options=[{'label': g, 'value': g} for g in sorted(df['gender'].unique())],
                        value=sorted(df['gender'].unique()),
                        placeholder="Gender"
                    ), width=3),
                    dbc.Col(dcc.RangeSlider(
                        id='sc_age',
                        min=int(df['AGE'].min()), max=int(df['AGE'].max()),
                        value=[int(df['AGE'].min()), int(df['AGE'].max())],
                        marks={i: str(i) for i in range(int(df['AGE'].min()), int(df['AGE'].max())+1, 10)},
                        tooltip={'placement':'bottom'},
                        updatemode='mouseup'
                    ), width=5),
                ], align="center", className="g-2")),
                dbc.CardBody(dcc.Graph(id='scat_graph', style={'height':'500px'}))
            ], className="mb-4"),
            
            # CAT Score Distribution
            html.H4("CAT Score Distribution"),
            dbc.Card(dbc.CardBody(dcc.Graph(id='hist_graph', style={'height':'400px'}))),
        ], className="p-4"),

        # Demographics Tab
        dbc.Tab(label="Demographics", tab_id="demographic", children=[
            # 6-Min Walk Distance by Severity
            html.H4("6-Min Walk Distance by Severity"),
            dbc.Card(dbc.CardBody(dcc.Graph(id='box_graph', style={'height':'400px'}))),

            # Average FEV1 by Severity
            html.H4("Average FEV1 by Severity"),
            dbc.Card(dbc.CardBody(dcc.Graph(id='bar_graph', style={'height':'400px'}))),

            # Gender Split
            html.H4("Gender Split"),
            dbc.Card(dbc.CardBody(dcc.Graph(id='pie_graph', style={'height':'400px'}))),
        ], className="p-4"),

        # Global Impact Tab
        dbc.Tab(label="Global Impact", tab_id="global", children=[
            html.Div([
                html.H3("WHO Data Analysis", className="text-center mb-4"),
                
                dbc.Alert([
                    html.H4("WHO Data Not Found", className="alert-heading"),
                    html.P("The WHO mortality dataset was not found in your data folder. "
                          "This tab requires the WHO.csv file to display global COPD mortality analysis.")
                ], color="warning", className="mb-4") if not has_who_data else None,
                
                # WHO Data Visualizations (Conditionally shown)
                html.Div([
                    html.H4("COPD Mortality Rate by Region"),
                    dcc.Graph(id='region_mortality', figure=px.bar(
                        who_df.groupby('Region Name').agg({'Death rate per 100 000 population': 'mean'}).reset_index(),
                        x='Region Name', y='Death rate per 100 000 population',
                        title="Average COPD Mortality Rate by Region",
                        template='plotly_white'
                    ) if has_who_data else {}),
                    
                    html.H4("Global COPD Mortality Trend"),
                    dcc.Graph(id='global_trend', figure=px.line(
                        who_df.groupby('Year').agg({'Number': 'sum'}).reset_index(),
                        x='Year', y='Number',
                        title="Global Trend of COPD Deaths Over Time",
                        template='plotly_white'
                    ) if has_who_data else {}),
                ]) if has_who_data else None,
            ]),
        ], className="p-4"),

        # Advanced Analysis Tab
        dbc.Tab(label="Advanced Analysis", tab_id="advanced", children=[
            # Linear Regression Coefficients
            html.H4("Linear Regression Coefficients (Numeric Features Only)"),
            
            # Get coefficients from a simple Linear Regression
            html.Div([
                dcc.Graph(id='coef_graph', figure=px.bar(
                    pd.DataFrame({
                        'feature': numeric_cols,
                        'coefficient': LinearRegression().fit(
                            numeric_pipe.fit_transform(X_train[numeric_cols]), 
                            y_train_enc
                        ).coef_
                    }).sort_values('coefficient', ascending=False),
                    x='coefficient', y='feature', 
                    orientation='h',
                    title='Linear Model Coefficients (Numeric Features Only)',
                    template='plotly_white'
                ).update_layout(height=500, yaxis={'categoryorder':'total ascending'}))
            ]),
            
            # PackHistory Distribution
            html.H4("PackHistory Distribution"),
            dbc.Card([
                dbc.CardHeader(dbc.Row([
                    dbc.Col(html.Span("Age Range:", style={'fontWeight':'bold'}), width="auto"),
                    dbc.Col(dcc.RangeSlider(
                        id='pack_age',
                        min=int(df['AGE'].min()), max=int(df['AGE'].max()),
                        value=[int(df['AGE'].min()), int(df['AGE'].max())],
                        marks={i: str(i) for i in range(int(df['AGE'].min()), int(df['AGE'].max())+1, 10)},
                        tooltip={'placement':'bottom'}
                    ), width=10),
                ], align="center", className="g-2")),
                dbc.CardBody(dcc.Graph(id='pack_graph', style={'height':'400px'}))
            ], className="mb-5"),
            
        ], className="p-4"),
    ], className="nav-fill"),
    
    # Footer
    html.Footer(
        html.P("COPD Dashboard created with Dash | Data sources: Patient dataset & WHO Mortality Database", 
            className="text-center text-muted mt-5")
    )
])

# --- Callbacks ---
@app.callback(
    Output('scat_graph','figure'),
    Input('sc_sev','value'),
    Input('sc_gen','value'),
    Input('sc_age','value'),
)
def update_scatter(sev, gen, age):
    dff = filter_df(df, sev, gen, age)
    return px.scatter(dff, x='FEV1', y='MWT1Best',
                      color='COPDSEVERITY', symbol='gender',
                      template='plotly_white')

@app.callback(Output('hist_graph','figure'), Input('hist_graph','id'))
def update_hist(_):
    return px.histogram(df, x='CAT', color='COPDSEVERITY',
                        barmode='group', nbins=20,
                        template='plotly_white')

@app.callback(Output('box_graph','figure'), Input('box_graph','id'))
def update_box(_):
    return px.box(df, x='COPDSEVERITY', y='MWT1Best',
                  points='all', template='plotly_white')

@app.callback(Output('bar_graph','figure'), Input('bar_graph','id'))
def update_bar(_):
    df2 = df.groupby('COPDSEVERITY', as_index=False)['FEV1'].mean()
    return px.bar(df2, x='COPDSEVERITY', y='FEV1', template='plotly_white')

@app.callback(Output('pie_graph','figure'), Input('pie_graph','id'))
def update_pie(_):
    cnt = df['gender'].value_counts().reset_index()
    cnt.columns = ['gender','count']
    return px.pie(cnt, names='gender', values='count', template='plotly_white')

@app.callback(Output('pack_graph','figure'), Input('pack_graph','id'))
def update_pack(_):
    return px.histogram(df, x='PackHistory', nbins=20, template='plotly_white')

# --- Run server ---
if __name__ == '__main__':
    app.run(debug=True, port=8050)