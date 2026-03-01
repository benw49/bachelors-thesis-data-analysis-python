import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

def calculate_co2_costs(co2_df: pd.DataFrame, training_df: pd.DataFrame, cols):
    #calculate total inference emissions for a scenario where there are an different amounts of inferences
    # *assuming that an average user does between 300 to 3,000 inferences per month*
    # *assuming also that said user has a Hugging Face style setup, i.e. they use 8 H100 GPUs for inference and are located
    # in Virginia* 
    # *and assuming different ranges, that 15-40% of downloaders are Daily Active Users (DAU), since not every download may be an active user*
    #then convert total emissions from kg to tons
    DAU = [0.15,0.2,0.3,0.4]
    inferences = [300,750,3000]
    inference_labels = get_inference_labels(inferences)

    total_co2_emissions_lower = []
    total_co2_emissions_upper = []

    for user_scaler in DAU:
        for I in inferences: 
            co2_emissions_users_lower = ((co2_df['Average CO2 cost (kg) per prompt'] * I * user_scaler * co2_df['downloads']) / 1000).sum() * 66
            co2_emissions_users_upper = ((co2_df['Average CO2 cost (kg) per prompt'] * I * user_scaler * co2_df['downloads']) / 1000).sum() * 200
            total_co2_emissions_lower.append(co2_emissions_users_lower)
            total_co2_emissions_upper.append(co2_emissions_users_upper)

    #plot the social cost of estimated monthly co2 emissions and place the bar charts next to each other, 
    #making sure that they do not overlap 
    x = np.arange(len(inference_labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, total_co2_emissions_lower, width=width, label='Lower bound ($66)', color='red', edgecolor='black')
    plt.bar(x + width/2, total_co2_emissions_upper, width=width, label='Upper bound ($200)', color='blue',edgecolor='black')

    plt.xticks(x, inference_labels,rotation=45,ha='right')
    plt.xlabel('DAU Scenarios')
    plt.ylabel('Social cost of total Emissions (in USD)')
    plt.title('Social cost of estimated total monthly carbon emissions for DAU (daily active user) scenarios')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def calculate_water_consumption(energy_df: pd.DataFrame):
    #WUE and PUE values assuming HuggingFace likely used AWS us-east-1 servers in Northern Virginia
    onsiteWUE = 0.12
    offsiteWUE = 0.63
    PUE = 1.15

    energy_df['Average water consumption (L) per prompt'] = energy_df['Average Energy Cost (kWh) per prompt'] * (onsiteWUE+(PUE*offsiteWUE))

    inferences = [300,750,3000]
    DAU = [0.15,0.2,0.3,0.4]
    inference_labels = get_inference_labels(inferences)
    total_water_consumption = []

    for user_scaler in DAU:
        for I in inferences:
            water_consumption = (energy_df['Average water consumption (L) per prompt'] * I * user_scaler * energy_df['downloads']).sum()
            total_water_consumption.append(water_consumption)

    #plot barchart and shift labels by 45 degrees to enhance readability, and set y-axis scale to 10^7
    plt.figure(figsize=(8,15))
    plt.bar(inference_labels,total_water_consumption,color='blue',width=0.5)    
    plt.tick_params(axis='x',labelsize=8)
    plt.xticks(rotation=45,ha='right')
    plt.title('Total monthly water consumption (L) for daily active user scenarios')
    plt.xlabel('DAU Scenarios')
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.gca().yaxis.get_major_formatter().set_powerlimits((7, 7))
    plt.ylabel('Total Water Consumption (L)')
    plt.show()

#function for finding out which of the models form the datasets can be found in the downloads data
def find_alike_models(cols, df1: pd.DataFrame, df2: pd.DataFrame):
    mask = df2['fullname'].isin(df1['fullname'])
    df_filtered = df2.loc[mask,cols]
    df_filtered_final = pd.merge(df_filtered,df1,on="fullname")
    return df_filtered_final

#function that gets the inference labels in the proper order for graphing on a bar chart
def get_inference_labels(inferences):
    total_water_consumption = []
    scenarios_labels = ['15%','20%','30%','40%']
    inference_categories = ['300 inferences', '750 inferences', '3000 inferences']

    inference_labels = []

    for scenario in scenarios_labels:
        for category in inference_categories:
            inference_labels.append(f'{scenario},{category}')

    return inference_labels
    
def clean_inference_data():
    #import csv files
    downloads_data = pd.read_csv("top-models-by-downloads.csv")
    leaderboard_data = pd.read_csv("openllm_leaderboard.csv")
    training_data = pd.read_csv("carbon_training_data.csv")

    downloads_data.rename(columns={"Model":"fullname"}, inplace=True)
    training_data.rename(columns={"Hugging Face Model Name":"fullname"}, inplace=True)
    
    co2_df = find_alike_models(["fullname", "CO2 cost (kg)"], downloads_data, leaderboard_data)

    #create a new column in the main (cleaned) dataset 
    #this represents the average co2 cost per prompt for hugging face's analysis, since they ran benchmarks
    #that totaled 21,682 prompts of varying lengths and complexities in their openLLM leaderboard
    co2_df['Average CO2 cost (kg) per prompt'] = co2_df['CO2 cost (kg)'] / 21682
    co2_df['Average Energy Cost (kWh) per prompt'] = (co2_df['CO2 cost (kg)'] * 1000)/(269.8*21682)

    calculate_co2_costs(co2_df,training_data,["fullname", "CO2 cost (kg)"])
    calculate_water_consumption(co2_df)