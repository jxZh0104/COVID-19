#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyreadr
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize
from mpl_toolkits.basemap import Basemap
from matplotlib import animation, rc
from IPython.display import HTML, Image
import plotly.express as px
import plotly.graph_objects as go
import pycountry
import json


# reference: http://louistiao.me/posts/notebooks/save-matplotlib-animations-as-gifs/
# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')


# In[2]:


result = pyreadr.read_r("COVID-19_cleaned_cases.RData")
result.keys()


# In[3]:


# data extraction and cleaning
global_counts_total = result['total_counts_global']
us_counts_total = result['total_counts_us']
global_counts_new = result['new_counts_global']
us_counts_new = result['new_counts_us']

def df_tidy(df):
    ncol = len(df.columns)
    nrow = len(df.index)
    df_new = df[list(df.columns[-3:])+list(df.columns[:-3])]
    df_dict = {}
    for i in range(nrow):
        df_dict[i] = df_new[df_new.columns[0]][i]
    df_new.rename(index=df_dict, inplace = True)
    return df_new

global_counts_total = df_tidy(global_counts_total)
us_counts_total = df_tidy(us_counts_total)
global_counts_new = df_tidy(global_counts_new)
us_counts_new = df_tidy(us_counts_new)


# In[11]:


global_counts_origin = pd.read_csv('jhu/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
                                  header = 0)
china_counts = global_counts_origin.loc[global_counts_origin['Country/Region'] == 'China']
china_counts


# In[4]:


#remove NaN rows
def remove_rows(df):
    rownames = df.index
    removed_indices = []
    for i in range(len(df)):
        name = df.index
        if np.isnan(df["lat"][i]) or np.isnan(df["lon"][i]):
            removed_indices.append(rownames[i])
    return df.drop(removed_indices, axis = 0)

global_counts_total_geo = remove_rows(global_counts_total)
us_counts_total_geo = remove_rows(us_counts_total)
global_counts_new_geo = remove_rows(global_counts_new)
us_counts_new_geo = remove_rows(us_counts_new)

def remove_geo(df):
    df_new = df[df.columns[3:]]
    return df_new

global_counts_total = remove_geo(global_counts_total)
us_counts_total = remove_geo(us_counts_total)
global_counts_new = remove_geo(global_counts_new)
us_counts_new = remove_geo(us_counts_new)


# In[5]:


# reference: https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/
def prediction_interval(model, params, X_train, y, number_new, alpha = 0.05, freq = None):
    ''' Compute a prediction interval around the model's prediction of 'number_new' new dates.
    INPUT
        model: function that takes X and params as parmaeters
        params: paramaters returned by curve_fit()
        X_train: Series Index (row names) from a dataframe
        y: Series Data (one row) from a dataframe
        number_new: number of new data points
        alpha: float = 0.05 
            The prediction uncertainty
        freq: freq param of pd.to_datetime() method

    OUTPUT
        A list of triples (`lower`, `pred`, `upper`) in the form of a number_new * 3 dataframe
        with `pred` being the prediction  of the model 
        and `lower` and `upper` constituting the lower- and upper 
        bounds for the prediction interval around `pred`, respectively. 
    '''

    # Number of training samples
    n = len(X_train)
    X = np.arange(n)
    x_new = np.array(range(n, n+number_new))
    # The authors choose the number of bootstrap samples as the square root 
    # of the number of samples
    nbootstraps = np.sqrt(n).astype(int)

    # Compute the m_i's and the validation residuals
    bootstrap_preds, val_residuals = np.zeros((nbootstraps,number_new)), []
    for b in range(nbootstraps):
        bootstrap_b = np.zeros((number_new))
        x_trains = np.random.choice(range(n), size = n, replace = True)
        x_vals = np.array([idx for idx in range(n) if idx not in x_trains])
        preds = model(x_vals, params[0], params[1], params[2])
        val_residuals.append(y[x_vals] - preds)
        bootstrap_preds[b, :] = model(x_new, params[0], params[1], params[2])
    bootstrap_preds -= np.mean(bootstrap_preds, axis = 0, keepdims = True)
    val_residuals = np.concatenate(val_residuals)

    # Compute the prediction and the training residuals
    preds = model(X, params[0], params[1], params[2])
    train_residuals = y - preds

    # Take percentiles of the training- and validation residuals to enable 
    # comparisons between them
    val_residuals = np.percentile(val_residuals, q = np.arange(100))
    train_residuals = np.percentile(train_residuals, q = np.arange(100))

    # Compute the .632+ bootstrap estimate for the sample noise and bias
    no_information_error = np.mean(np.abs(np.random.permutation(y) - np.random.permutation(preds)))
    generalisation = np.abs(val_residuals - train_residuals)
    no_information_val = np.abs(no_information_error - train_residuals)
    relative_overfitting_rate = np.mean(generalisation / no_information_val)
    weight = .632 / (1 - .368 * relative_overfitting_rate)
    residuals = (1 - weight) * train_residuals + weight * val_residuals

    # Construct the C set and get the percentiles
    pred_intervals = np.zeros((number_new, 3))
    qs = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    for i in range(number_new):
        C = np.array([m + o for m in bootstrap_preds[i, ] for o in residuals])
        percentiles = np.percentile(C, q = qs)
        pred = model(x_new[i], params[0], params[1], params[2])
        pred_intervals[i, :] = [pred + percentiles[0], pred, pred + percentiles[1]]
    
    # convert numpy array to panda df
    New_Dates = pd.date_range(start = X_train[0], periods=number_new+n,freq=freq)
    pred_df = pd.DataFrame({'lower': pred_intervals[:,0],
                           'pred': pred_intervals[:,1],
                           'upper': pred_intervals[:,2]},
                          index = New_Dates[n:])
    return pred_df
    


# In[6]:


# reference: https://medium.com/analytics-vidhya/how-to-predict-when-the-covid-19-pandemic-will-stop-in-your-country-with-python-d6fbb2425a9f
def logistic(X, c, k, m):
    return c/(1+np.exp(-k*(X-m)))

def gaussian(X, mu, sigma, c):
    return c*np.exp(-1/2*((X-mu)/sigma)**2)

def curve_fit(dataset, name, choice):
    '''
    fits the confirmed cases of a specific region given by 'name' (which are index names) in the 'dataset'
    the 'choice' has to be either total confirmed cases ('total') up to rach date or daily new cases ('new')
    '''
    X = np.arange(len(dataset.loc[name]))
    y = dataset.loc[name].values
    if choice == "total":
        logistic_model, cov = optimize.curve_fit(logistic,
                                xdata=X, 
                                ydata=y, 
                                maxfev=10000,
                                p0=[np.max(X), 1, 1])
        return logistic_model
    else:
        gaussian_model, cov = optimize.curve_fit(gaussian,
                                xdata=X, 
                                ydata=y, 
                                maxfev=10000,
                                p0=[np.mean(X), 1, 1])
        return gaussian_model
    
def forecast(dataset, region, name, choice, zoom = 30, figsize = (20, 5), pred_ahead = 0, freq = None):
    '''
    this function both outputs a plot and returns a dataframe with 3 columns:
    'date', 'confirmed cases' and 'predicted cases';
    region: 'global'/'US'
    model_params are those that are returned by curve_fit(),
    pred_ahead is the number of days to be forecasted (must be > 0)
    freq is the frequency by which data wish to be presented.
    zoom: how many last datas what to be presented in the plot;
        e.g. zoom = 30 : plot the last 30 of the original data plus the new predicted data
    the other parameters are the same as in prediction_interval
    '''
    current_dates = dataset.loc[name].index
    y = dataset.loc[name]
    New_Dates = pd.date_range(start=current_dates[0],periods=pred_ahead+len(current_dates),freq=freq)
    model_params = curve_fit(dataset, name, choice)
    X = np.arange(len(New_Dates))
    if choice == "total":
        fitted_y = logistic(X, model_params[0], model_params[1], model_params[2])
        prediction = prediction_interval(logistic, model_params,current_dates, y, number_new = pred_ahead, freq=freq)
    else:
        fitted_y = gaussian(X, model_params[0], model_params[1], model_params[2])
        prediction = prediction_interval(gaussian, model_params,current_dates, y, number_new = pred_ahead, freq=freq)
        
    df_new = pd.DataFrame(['NaN']*(pred_ahead), index = New_Dates[len(current_dates):])
    df_confirmed = pd.concat([dataset.loc[name],df_new])
    df = pd.DataFrame({'Dates': New_Dates, 'Confirmed Cases': df_confirmed[0], 'Predicted Cases': fitted_y})
    dict_df = {}
    for i in range(len(New_Dates)):
        dict_df[df.index[i]] = i
    df.rename(index = dict_df, inplace = True)
    
    # plotting with prediction_intervals
    fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize=figsize)
    line1 = ax[0].plot(New_Dates, fitted_y, label = 'Predicted Cases', color = 'red')
    line2 = ax[1].plot(New_Dates[(len(current_dates)-zoom):], fitted_y[(len(current_dates)-zoom):], color = 'red', label = 'Predicted Cases')
    line3 = prediction['pred'].plot(ax=ax[0], grid=False, color="red")
    ax[0].fill_between(x=prediction.index, y1=prediction['lower'], y2=prediction['upper'], color='b', alpha=0.3)
    line4 = prediction["pred"].plot(ax=ax[1], grid=False, color="red")
    ax[1].fill_between(x=prediction.index, y1=prediction['lower'], y2=prediction['upper'], color='b', alpha=0.3)
    
    if choice == "total":
        line5 = ax[0].plot(pd.to_datetime(current_dates), dataset.loc[name], 'ro', markersize = 2, label = 'Confirmed Cases', color = 'blue')
        ax[0].legend(line1 + line5, ['Predicted Cases', 'Confirmed Cases'], loc = 'best')
        line6 = ax[1].plot(pd.to_datetime(current_dates)[(len(current_dates)-zoom):], dataset.loc[name][(len(current_dates)-zoom):], markersize = 2, label = 'Confirmed Cases')
        ax[1].legend(line2 + line6, ['Predicted Cases', 'Confirmed Cases'], loc = 'best')
    else:
        # create bar plots
        ax2 = ax[0].twinx()
        ax2.set_ylim(ax[0].get_ylim()) # so that the bar plot and the line graph habe the dame scale
        ax3 = ax[1].twinx()
        ax3.set_ylim(ax[1].get_ylim()) # so that the bar plot and the line graph habe the dame scale
        ax2.bar(pd.to_datetime(current_dates), dataset.loc[name], align='center', alpha = 0.5, width = 0.5, label = 'Confirmed Cases', color = 'blue')
        ax2.grid(b=False)  
        lines, labels = ax[0].get_legend_handles_labels()
        line, label = [lines[0]], [labels[0]] # make sure they are lists, do not retrieve the prediction interval line and label
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(line + lines2, label + labels2, loc='upper left', prop={'size': 20}) 
        ax3.bar(pd.to_datetime(current_dates)[(len(current_dates)-zoom):], dataset.loc[name][(len(current_dates)-zoom):], align='center', alpha = 0.5, width = 0.5, label = 'Confirmed Cases', color = 'blue')
        ax3.grid(b=False)  
        lines1, labels1 = ax[1].get_legend_handles_labels()
        line_1, label_1 = [lines1[0]], [labels1[0]]
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax3.legend(line_1 + lines3, label_1 + labels3, loc='upper left', prop={'size': 20}) 
        
    fig.suptitle('Forecast Plot of COVID-19 Cases', fontsize = 20)
    fig.text(0.27, 0.05, 'Date', fontsize = 15)
    fig.text(0.73, 0.05, 'Date', fontsize = 15)
    fig.text(0.07, 0.35, 'Number of ' + choice.title() + ' Cases', size = 15, rotation = 90)
    plt.savefig('Forecast_Plot_' + region.title() + '_' + choice.title() + '.png')
    plt.show()
    return df
    


# In[86]:


# Test Module
params = curve_fit(global_counts_total, 'US', 'total')
x = global_counts_total.loc['US'].index
y = global_counts_total.loc['US']
prediction_interval(logistic, params, x, y, 10)


# In[11]:


# Test Module
forecast(global_counts_total, 'global', 'US', 'total', zoom = 30, pred_ahead = 10, freq = None)


# In[88]:


# Test Module
forecast(us_counts_new, 'US', 'California', 'new', zoom = 30, pred_ahead = 10, freq = None)


# In[6]:


# https://makersportal.com/blog/2018/7/20/geographic-mapping-from-a-csv-file-using-python-and-basemap
# obsolete function
def map_plot(dataset, date):
    '''
    this function generates a plot showing the confirmed cases on the specified date (a string object) and 
    within a given dataset (a pd dataframe with "lat" and "lon" columns)
    '''
    font = {'family' : 'verdana',
            'size'   : 16}
    matplotlib.rc('font', **font)

    lats,lons,names,count = dataset["lat"], dataset["lon"], dataset.index, dataset[date]

    # How much to zoom from coordinates (in degrees)
    zoom_scale = 3

    # Setup the bounding box for the zoom and bounds of the map
    bbox = [np.min(lats)-zoom_scale,np.max(lats)+zoom_scale,            np.min(lons)-zoom_scale,np.max(lons)+zoom_scale]

    fig, ax = plt.subplots(figsize=(12,7))
    plt.title("COVID-19 Cases " + date)
    # Define the projection, scale, the corners of the map, and the resolution.
    m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],                llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10)

    # Draw coastlines and fill continents and water with color
    m.drawcoastlines()
    m.fillcontinents(color='#CCCCCC',lake_color='lightblue')

    # draw parallels, meridians, and color boundaries
    m.drawparallels(np.arange(bbox[0],bbox[1],(bbox[1]-bbox[0])/5),labels=[1,0,0,0])
    m.drawmeridians(np.arange(bbox[2],bbox[3],(bbox[3]-bbox[2])/5),labels=[0,0,0,1],rotation=15)
    m.drawmapboundary(fill_color='lightblue')

    # format colors for elevation range
    count_min = np.min(dataset[dataset.columns[3]])
    count_max = np.max(np.max(dataset[dataset.columns[3:]]))
    cmap = plt.get_cmap('Reds')
    normalize = matplotlib.colors.LogNorm(vmin=1, vmax=count_max)

    # plot elevations with different colors using the numpy interpolation mapping tool
    # the range [50,200] can be changed to create different colors and ranges
    for ii in range(0,len(count)):
        x,y = m(lons[ii],lats[ii])
        color_interp = np.interp(count[ii],[count_min,count_max],[50,250])
        plt.plot(x,y,marker='o',markersize=6,color=cmap(int(color_interp)))

    # format the colorbar 
    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,norm=normalize,label='Confirmed Cases')

    # save the figure and show it
    # plt.savefig('asos_station_elevation.png', format='png', dpi=500,transparent=True)
    plt.show()


# In[98]:


# Test Module
map_plot(us_counts_total_geo, '8/30/20')


# In[7]:


# https://gist.github.com/rogerallen/1583593
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}


# In[8]:


# Map GIF
# https://towardsdatascience.com/covid-19-map-animation-with-python-in-5-minutes-2d6246c32e54
# https://towardsdatascience.com/how-to-create-an-animated-choropleth-map-with-less-than-15-lines-of-code-2ff04921c60b

def get_country_code(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

def plot_gif(dataset, region, choice, scale):
    '''
    this function displays an animated plot of the progress of COVID-19 cases over time
    dataset: a panda dataframe with the first 3 column as 'country'/'state', 'lat', and 'lon', 
            and the rest of the columns as dates
    region: 'global' or 'us'
    choice: 'total' or 'new'
    scale: 'continuous' or 'log'
    '''
    if region == 'global':
        dataset['iso_alpha_3'] = [get_country_code(name) for name in dataset['country']]
        df_long = pd.melt(dataset[[dataset.columns[0]]+list(dataset.columns[3:])], id_vars=['country','iso_alpha_3'])
        if scale == 'log':
            max_color_range = int(max(np.log10(df_long["value"])))
            fig = px.choropleth(df_long,                            # Input Dataframe
                         locations="iso_alpha_3",           # identify country code column
                         color=np.log10(df_long["value"]),  # identify representing column
                         hover_name="country",              # identify hover name
                         animation_frame="variable",        # identify date column
                         projection="natural earth",        # select projection
                         color_continuous_scale = 'Peach',  # select prefer color scale
                         range_color=(0, max_color_range), # select range of dataset
                         title = 'COVID-19 ' + choice.title() + ' Confirmed Cases (Global)'
                        )
            fig.update_layout(coloraxis_colorbar=dict(
                    tickvals=list(range(max_color_range)),
                    ticktext=[str(10**num) for num in range(max_color_range)],))
        else:
            fig = px.choropleth(df_long,                            
                         locations="iso_alpha_3",           
                         color="value",
                         hover_name="country",              
                         animation_frame="variable",        
                         projection="natural earth",        
                         color_continuous_scale = 'Peach',  
                         range_color=(0, 10**int(max(np.log10(df_long["value"])))), 
                         title = 'COVID-19 ' + choice.title() + ' Confirmed Cases (Global)'
                        )
        fig.show()          
        fig.write_html('global_map_' + choice + '.html')
    elif region == 'us':
        dataset['abbr'] = [us_state_abbrev[state] for state in dataset['state']]
        df_long = pd.melt(dataset[[dataset.columns[0]]+list(dataset.columns[3:])], id_vars=['state','abbr'])
        if scale == 'log': 
            max_color_range = int(max(np.log10(df_long["value"])))
            fig = px.choropleth(df_long, 
                  locations = 'abbr',
                  color=np.log10(df_long["value"]), 
                  animation_frame="variable",
                  color_continuous_scale="Peach",
                  locationmode='USA-states',
                  scope="usa",
                  range_color=(0, max_color_range),
                  title='COVID-19 Total Confirmed Cases (US)'
                 )
            fig.update_layout(coloraxis_colorbar=dict(
                    tickvals=list(range(max_color_range)),
                    ticktext=[str(10**num) for num in range(max_color_range)],))
        else:
            fig = px.choropleth(df_long, 
                  locations = 'abbr',
                  color="value", 
                  animation_frame="variable",
                  color_continuous_scale="Peach",
                  locationmode='USA-states',
                  scope="usa",
                  range_color=(0, 10**int(max(np.log10(df_long["value"])))),
                  title='COVID-19 ' + choice.title() + ' Confirmed Cases (US)'
                 )    
        fig.show()
        fig.write_html('us_map_' + choice + '.html')


# In[122]:


plot_gif(global_counts_total_geo, 'global', 'total', 'continuous')
plot_gif(us_counts_new_geo, 'us', 'new', 'continuous')


# In[9]:


data = json.load(open('CHN_adm1.json'))


# In[12]:


df_long_china = pd.melt(china_counts[[china_counts.columns[0]]+list(china_counts.columns[4:])], id_vars=['Province/State'])

df_long_china


# In[ ]:


fig = px.choropleth_mapbox(df_long_china, geojson=data, locations='Province/State', 
                           color='value',
                           animation_frame="variable",
                           color_continuous_scale="Viridis",
                           range_color=(0, 10**4),
                           mapbox_style="carto-positron",
                           zoom=3
                          )
fig.show()


# In[ ]:


china_counts


# In[ ]:




