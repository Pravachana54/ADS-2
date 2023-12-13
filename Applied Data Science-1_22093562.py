import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100



def dataframe_loading(filename):
    '''
    loading dataframe.
    '''
    data_year_feat = pd.read_excel(filename,engine="openpyxl")
    data_country_feat = pd.melt(data_year_feat, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                    var_name='Year', value_name='Value')
    data_country_feat = data_country_feat.pivot_table(index=['Year', 'Country Code', 'Indicator Name', 'Indicator Code'], columns='Country Name', values='Value').reset_index()
    data_country_feat = data_country_feat.drop_duplicates().reset_index()
    return data_year_feat,data_country_feat



data_year_feat,data_country_feat = dataframe_loading('world_bank_climate.xlsx')



def data_yearly(data,m,n,o):
    data_to_get = data.copy()
    years_need=[i for i in range(m,n,o)]
    req_feat=['Country Name','Indicator Name']
    req_feat.extend(years_need)
    data_to_get =  data_to_get[req_feat]
    data_to_get = data_to_get.dropna(axis=0, how="any")
    return data_to_get


sample_data = data_yearly(data_year_feat,1990,2020,4)



countries_for_analysis = sample_data['Country Name'].value_counts().index.tolist()[30:40]


def feature_value_checked(data,column,values):
    data_rand= data.copy()
    data_rand= data_rand[data_rand[column].isin(values)].reset_index(drop=True)
    return data_rand



sample_data_country  = feature_value_checked(sample_data,'Country Name',countries_for_analysis)


country_dict = dict()
for i in range(sample_data_country.shape[0]):
    if sample_data_country['Country Name'][i] not in country_dict.keys():
        country_dict[sample_data_country['Country Name'][i]]=[sample_data_country['Indicator Name'][i]]
    else:
        country_dict[sample_data_country['Country Name'][i]].append(sample_data_country['Indicator Name'][i])
    

for k,v in country_dict.items():
    country_dict[k] = set(v)



inter = country_dict['United Arab Emirates']
for v in country_dict.values():
    inter = inter.intersection(v)



print(sample_data_country.describe())


df_no2 = feature_value_checked(sample_data_country,'Indicator Name',['Nitrous oxide emissions (thousand metric tons of CO2 equivalent)'])


print(df_no2.describe())

df_no2  = feature_value_checked(df_no2,'Country Name',countries_for_analysis)


def countries_bar_plot(data,indicator_variable):
    df_bar = data.copy()
    df_bar.set_index('Country Name', inplace=True)
    num_to_check = df_bar.columns[df_bar.dtypes == 'float64']
    df_bar = df_bar[num_to_check]
    plt.figure(figsize=(50, 50))
    df_bar.plot(kind='bar')
    plt.title(indicator_variable)
    plt.xlabel('Country Name')    
    plt.legend(title='Year', bbox_to_anchor=(1.10, 1), loc='upper left')
    plt.show()


countries_bar_plot(df_no2,'Nitrous oxide emissions (thousand metric tons of CO2 equivalent)')

df_ren =  feature_value_checked(sample_data_country,'Indicator Name',['Renewable energy consumption (% of total final energy consumption)'])


print(df_ren.describe())

countries_bar_plot(df_ren,'Renewable energy consumption (% of total final energy consumption)')


df_maur= feature_value_checked(sample_data_country,'Country Name',['Mauritania'])


def indicator_selected_data(data):
    df_selected=data.copy()
    # Melt the DataFrame
    df_selected = df_selected.melt(id_vars='Indicator Name', var_name='Year', value_name='Value')

    # Pivot the DataFrame
    df_selected = df_selected.pivot(index='Year', columns='Indicator Name', values='Value')

    # Reset index
    df_selected.reset_index(inplace=True)
    df_selected = df_selected.apply(pd.to_numeric, errors='coerce')
    del df_selected['Year']
    df_selected = df_selected.rename_axis(None, axis=1)
    return df_selected

    

df_maur= indicator_selected_data(df_maur)


features_to_select = ['Forest area (% of land area)',
 'Methane emissions (kt of CO2 equivalent)',
                     'Agricultural land (% of land area)',
 'Nitrous oxide emissions (thousand metric tons of CO2 equivalent)',
 'Renewable energy consumption (% of total final energy consumption)',
                     'CO2 emissions (metric tons per capita)',
 'Urban population growth (annual %)']



df_maur = df_maur[features_to_select]


print(df_maur.corr())



sns.heatmap(df_maur.corr(), annot=True, cmap='YlGnBu', linewidths=.5, fmt='.3g')


df_co2_em= feature_value_checked(sample_data_country,'Indicator Name',['CO2 emissions (metric tons per capita)'])

print(df_co2_em.describe())


df_urban= feature_value_checked(sample_data_country,'Indicator Name',['Urban population growth (annual %)'])


#df_urban.to_excel("ub6_assign.xlsx")



def time_series_graph(data,indicator_label):
    df_selected = data.copy()
    df_selected.set_index('Country Name', inplace=True)
    num_check = df_selected.columns[df_selected.dtypes == 'float64']
    df_selected = df_selected[num_check]

    plt.figure(figsize=(12, 6))
    for count in df_selected.index:
        plt.plot(df_selected.columns, df_selected.loc[count], label=count, linestyle='dashed', marker='o')

    plt.title(indicator_label)
    plt.xlabel('Year')
    plt.legend(title='Country', bbox_to_anchor=(1.20, 1), loc='upper left')

    plt.show()



time_series_graph(df_co2_em,'CO2 emissions (metric tons per capita)')


df_met= feature_value_checked(sample_data_country,'Indicator Name',['Methane emissions (kt of CO2 equivalent)'])



print(df_met.describe())


time_series_graph(df_met,'Methane emissions (kt of CO2 equivalent)')



df_year_col_uae = feature_value_checked(sample_data_country,'Country Name',['United Arab Emirates'])
df_year_col_uae = indicator_selected_data(df_year_col_uae)
df_year_col_uae = df_year_col_uae[features_to_select]
sns.heatmap(df_year_col_uae.corr(), annot=True, cmap='YlGnBu', linewidths=.5, fmt='.3g')



df_year_col_arg= feature_value_checked(sample_data_country,'Country Name',['Argentina'])
df_year_col_arg = indicator_selected_data(df_year_col_arg)
df_year_col_arg = df_year_col_arg[features_to_select]
plt.figure()
sns.heatmap(df_year_col_arg.corr(), annot=True, cmap='YlGnBu', linewidths=.5, fmt='.3g')



