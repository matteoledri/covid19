import streamlit as st
import pandas as pd 
import numpy as np 


####### Utility methods
def load_protezione_civile_data_italy(addPopulation = False, populationFileName = 'population_ita_regions.csv'):

    fileName = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-json/dpc-covid19-ita-regioni.json'

    df = pd.read_json(fileName)    
    
    df=df.rename(columns={'lat':'Lat','long':'Long','data':'Date','stato':'Country','denominazione_regione':'State/Region',                          'totale_casi':'Confirmed','deceduti':'Deaths','dimessi_guariti':'Recovered'})
    
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S', utc=True)
    if addPopulation:
        rbdf = load_population_italian_regions(fileName = populationFileName)
        df = df.merge(rbdf, on=['State/Region'])
    return df


def load_who_data_world(addPopulation = False, populationFileName = 'population_world_countries.csv'):
    confirmed = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv",keep_default_na=False)
    deaths = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv",keep_default_na=False)
    recovered = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv",keep_default_na=False)

    confirmed["Case_Type"] = "Confirmed"
    deaths["Case_Type"] = "Deaths"
    recovered["Case_Type"] = "Recovered"

    key_columns = ["Country/Region",
                   "Province/State",
                   "Lat",
                   "Long",
                   "Case_Type"]

    data = [confirmed, deaths, recovered]

    def unpivot(df):
        # unpivot all non-key columns
        melted = df.melt(id_vars=key_columns, var_name="Date", value_name="Cases")
        # change our new Date field to Date type
        melted["Date"]= pd.to_datetime(melted["Date"]) 

        return melted

    unpivoted_data = list(map(unpivot, data))

    dfConfirmed = unpivoted_data[0]
    dfDeaths    = unpivoted_data[1]
    dfRecovered = unpivoted_data[2]

    dfConfirmed = dfConfirmed.drop(columns=['Case_Type'])
    dfConfirmed = dfConfirmed.rename(columns={"Cases": "Confirmed"})
    dfDeaths    = dfDeaths.drop(columns=['Case_Type'])
    dfDeaths    = dfDeaths.rename(columns={"Cases": "Deaths"})
    dfRecovered = dfRecovered.drop(columns=['Case_Type'])
    dfRecovered = dfRecovered.rename(columns={"Cases": "Recovered"})

    df = dfConfirmed.merge(dfDeaths, on=['Country/Region','Province/State','Lat','Long','Date'])
    df = df.merge(dfRecovered, on=['Country/Region','Province/State','Lat','Long','Date'])
    
    df = df.rename(columns={'Country/Region': 'Country', 'Province/State': 'State/Region'})
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', utc=True)
  
    if addPopulation:
        wbdf = load_population_wb(fileName = populationFileName)
        df = df.merge(wbdf, on=['Country'])

    return df

def load_population_wb(fileName = 'population_world_countries.csv'):
    import os
    if fileName and os.path.isfile(fileName):
        wbdf = pd.read_csv(fileName)
    else:    
        import wbdata
        import datetime

        wbdf = wbdata.get_dataframe({'SP.POP.TOTL':'Population'}, country='all', convert_date=True)
        wbdf = wbdf.reset_index()
        wbdf = wbdf.dropna()
        wbdf = wbdf.groupby(by=['country']).first()
        wbdf = wbdf.reset_index()
        wbdf = wbdf.rename(columns={'country':'Country'})
        wbdf = wbdf.drop(columns=['date'])
        #fix names to match the WHO datasource
        correctCountryNamesDict = {}
        oldNames = ["Brunei Darussalam", "Congo, Dem. Rep.", "Congo, Rep.", "Czech Republic",
          "Egypt, Arab Rep.", "Iran, Islamic Rep.", "Korea, Rep.", "St. Lucia", "West Bank and Gaza", "Russian Federation",
          "Slovak Republic", "United States", "St. Vincent and the Grenadines", "Venezuela, RB"]
        newNames = ["Brunei", "Congo (Kinshasa)", "Congo (Brazzaville)", "Czechia", "Egypt", "Iran", "Korea, South",
          "Saint Lucia", "occupied Palestinian territory", "Russia", "Slovakia", "US", "Saint Vincent and the Grenadines", "Venezuela"]
        for old,new in zip(oldNames,newNames):
            correctCountryNamesDict[old] = new
        wbdf = wbdf.replace({"Country": correctCountryNamesDict})
        # Data from wikipedia
        noDataCountries= pd.DataFrame({
          'Country':["Cruise Ship", "Guadeloupe", "Guernsey", "Holy See", "Jersey", "Martinique", "Reunion", "Taiwan*"],
          'Population':[3700, 395700, 63026, 800, 106800, 376480, 859959, 23780452]})

        wbdf = wbdf.append(noDataCountries).sort_values(by=['Country']).reset_index(drop=True)
        
        if fileName:
            wbdf.to_csv(fileName, index=False)
    
    return wbdf

def load_population_italian_regions(fileName = 'population_ita_regions.csv'):
    import os
    if os.path.isfile(fileName):
        rpdf = pd.read_csv(fileName)
        return rpdf
    else:
        print('The population file',fileName,'does not exist')
        return None

def processData(df, referenceCaseNumber=100, group='Country'):
    df = df.groupby(by=[group,'Date']).sum()

    df['deltaFromRefCases'] = np.abs(df['Confirmed']-referenceCaseNumber)

    modGroups = []

    for item in df.reset_index()[group].unique():
        dfGroup = df.loc[item]
        dfGroup = dfGroup.reset_index()
        dfGroup[group] = item
        referenceCasesDate = dfGroup.iloc[np.argmin(dfGroup['deltaFromRefCases'])]['Date']
        dfGroup['timeAfterRefCases'] = (dfGroup['Date']-referenceCasesDate).dt.days

        dfGroup['NewCases'] = dfGroup['Confirmed']-dfGroup['Confirmed'].shift(1)
        dfGroup['NewCasesIncrement'] = dfGroup['NewCases']/dfGroup['NewCases'].shift(1)
        dfGroup['GrowthRatePercent'] =  ((dfGroup['Confirmed']/dfGroup['Confirmed'].shift(1))-1.0)*100.0

        dfGroup['DeathRate'] = dfGroup['Deaths']/dfGroup['Confirmed']*100.0

        dfGroup['ConfirmedPer100k']=dfGroup['Confirmed']/dfGroup['Population']*1E5

        modGroups.append(dfGroup)

    df = pd.concat(modGroups)
    
    return df

def getLinePlot(data=None,xVar=None,yVars=[],colorVar=None, logX=False, logY=False, linetypeVar=None, tooltipVar=None,title='',axesSquare=False,height=300, lineWidth=1):

    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import colorlover as cl
    
    plotData = data.copy(deep=True)
    if xVar==None:
        plotData['plotIndex'] = plotData.index
        xVar='plotIndex'

    if colorVar==None:
        colorVar='cat'
        plotData.loc[:,colorVar] = ''
    plotData.loc[:,colorVar] = plotData.loc[:,colorVar].astype('category')

    if linetypeVar==None:
        linetypeVar='lineCat'
        plotData.loc[:,linetypeVar] = ''
    plotData.loc[:,linetypeVar] = plotData.loc[:,linetypeVar].astype('category')
    
    fig = make_subplots(rows=len(yVars), cols=1,
                            shared_xaxes=True, shared_yaxes=True,
                            vertical_spacing=0.01, print_grid=False)#, subplot_titles=yVars)
    
    categories=plotData[colorVar].cat.categories
    # colors = cl.scales['12']['qual']['Paired']*3
    colors = cl.scales['9']['qual']['Set1']*10

    linetypes=['solid','dot','dash','dashdot','longdash','longdashdot']*10
#     linetypes=['25px 0px 25px 0px','25px 25px 25px 25px']*10
    lineTypeCategories=plotData[linetypeVar].cat.categories
    
    
    if type(logY) is bool:
        logYList=[logY]*len(yVars)
    elif type(logY) is list:
        logYList = logY
    
    row = 1
    for yVar in yVars:
        for i in range(len(categories)):
            for j in range(len(lineTypeCategories)):
                subsetData = plotData[(plotData[colorVar]==categories[i]) & (plotData[linetypeVar]==lineTypeCategories[j])]
                
                showLegendGroupBool = True
                if row>1:
                    showLegendGroupBool = False
                trace = go.Scatter(
                    x=subsetData[xVar],
                    y=subsetData[yVar],
                    name = categories[i],
                    line = dict(color = (colors[i]),width = lineWidth, dash=linetypes[j]),
                    legendgroup = categories[i],
                    showlegend = showLegendGroupBool
                )
                if tooltipVar!=None:
                    trace.text=subsetData[tooltipVar]
                fig.append_trace(trace,row,1)
        fig.update_yaxes(title_text=yVar,row=row,col=1)
        if logYList[row-1]:
            fig.update_yaxes(type='log',row=row,col=1)
        if row==len(yVars):
            fig.update_xaxes(title_text=xVar,row=row,col=1)
        row = row + 1
        
    fig.update_layout(title=title)
    fig.update_layout(height=height*len(yVars))
    fig.update_layout(hovermode='closest')
    fig.update_layout(legend_orientation="h", legend_font_size=8)
    
    if logX:
        fig.update_layout(xaxis_type="log")
#     fig.update_layout(yaxis=dict(title=yVars[0]
#     fig.update_layout(xaxis=dict(title=xVar))
    
    if axesSquare:
        fig.update_layout(yaxis=dict(title=yVars[0],scaleanchor = "x"),width=height, height=height)
        
    return fig

##### end of Utility methods


st.sidebar.markdown('MG Covid Explorer')
itaWorldRadio = st.sidebar.radio(
    '',
     ('World', 'Italy'))

if itaWorldRadio=='World':
    groupVar='Country'
    df = load_who_data_world(addPopulation= True)
    defaultGroups = ['Italy','Spain','France','Germany','US','United Kingdom', 'New Zealand', 'China', 'Japan', 'Korea, South']
else:
    groupVar='State/Region'
    df = load_protezione_civile_data_italy(addPopulation=True)
    defaultGroups = ['Lombardia', 'Friuli Venezia Giulia', 'Marche', 'Sardegna', 'Emilia Romagna', 'Lazio']

refCasesStr = st.sidebar.text_input('Number of Cases for time offset', '100')
refCases=float(refCasesStr)



data = processData(df, referenceCaseNumber=refCases, group=groupVar)
availableGroups = list(data[groupVar].unique())


selectedGroups = st.sidebar.multiselect(
    label='Select '+groupVar,
    options=availableGroups,
    
    default=defaultGroups)

if len(selectedGroups)!=0:
    selectedData = data[data[groupVar].isin(selectedGroups)]
else:
    selectedData = data

figTimeSeries = getLinePlot(selectedData, xVar='timeAfterRefCases', yVars=['Confirmed','ConfirmedPer100k','NewCases','Deaths','Recovered','GrowthRatePercent','DeathRate'], colorVar=groupVar, logY=[True,False,False,True,True,False,False,False])


st.sidebar.markdown('Approved by ZuZu')



# st.text('This will appear first')
# # Appends some text to the app.

# my_slot1 = st.empty()
# # Appends an empty slot to the app. We'll use this later.

# my_slot2 = st.empty()
# # Appends another empty slot.

# st.text('This will appear last')
# # Appends some more text to the app.

# my_slot1.text('This will appear second')
# # Replaces the first empty slot with a text string.

st.write(figTimeSeries)
# # Replaces the second empty slot with a chart.

st.dataframe(data=selectedData.groupby(groupVar).last())