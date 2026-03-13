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

    sf_ny_roundtrip_flights_co2_tons_per_pass = 180.4
    czech_residents_emissions_tons_per_capita = 7.04

    opp_cost_co2_emissions_flights = []
    opp_cost_co2_emissions_czech_residents = []

    actual_co2_emissions = []

    for i in total_co2_emissions_lower:
        opp_cost_co2_emissions_flights.append((i/66)/sf_ny_roundtrip_flights_co2_tons_per_pass)
        opp_cost_co2_emissions_czech_residents.append((i/66)/czech_residents_emissions_tons_per_capita)
        actual_co2_emissions.append(i/66)

    #plot the social cost of estimated monthly co2 emissions and place the bar charts next to each other, 
    #making sure that they do not overlap 
    #also plotting for the opportunity costs of co2 emissions
    plt.rcParams['axes.grid'] = False
    x = np.arange(len(inference_labels))
    width = 0.35

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6))
    ax1.bar(x - width/2, total_co2_emissions_lower, width=width, label='Lower bound ($66)', color='red', edgecolor='black')
    ax1.bar(x + width/2, total_co2_emissions_upper, width=width, label='Upper bound ($200)', color='blue',edgecolor='black')

    ax1.set_xticks(x,inference_labels,rotation=45,ha='right')
    ax1.set_xlabel('DAU Scenarios')
    ax1.set_ylabel('Social cost of total Emissions (in USD)')
    ax1.set_title('Social cost of estimated total \n monthly carbon emissions \n for DAU (daily active user) scenarios')
    ax1.legend()

    ax2.bar(x,actual_co2_emissions, width=width,color='blue',edgecolor='black')
    ax2.set_ylabel('CO2 emissions (tCO2eq)')
    ax2.set_title('Total monthly CO2 emissions (tCO2eq) emissions per DAU scenario')
    ax2.set_xlabel('DAU Scenarios')
    ax2.set_xticks(x,inference_labels,rotation=45,ha='right')

    #plotting opportunity costs of carbon emissions for the DAU scenarios
    ax3.bar(x,opp_cost_co2_emissions_flights,width=width,color='green',edgecolor='black')
    ax3.set_ylabel('Number of SFO to JFK flights')
    ax3.set_xlabel('DAU Scenarios')
    ax3.set_title('Opportunity costs of inference scenarios \n (SFO to JFK flights)')
    ax3.set_xticks(x,inference_labels,rotation=45,ha='right')

    ax4.bar(x,opp_cost_co2_emissions_czech_residents,width=width,color='orange',edgecolor='black')
    ax4.set_ylabel('Number of czech households')
    ax4.set_xlabel('DAU Scenarios')
    ax4.set_title('Opportunity costs of inference scenarios \n (number of czech households)')
    ax4.set_xticks(x,inference_labels,rotation=45,ha='right')

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
    plt.rcParams['axes.grid'] = False
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

def proprietary_model_co2():
    #use the numbers for chatGPT to plot the data with
    #2.5 billion prompts per day, 0.34 watt-hours of energy per average prompt
    prompts_per_month_gpt = (2.5e+9) * 30 
    energy_per_prompt_kWh_gpt = 0.34 / 1000 
    carbon_grid_intensity_grams_gpt = 384 
    co2_total_per_month_gpt_tons = ((prompts_per_month_gpt * energy_per_prompt_kWh_gpt * carbon_grid_intensity_grams_gpt)/1e+6)
    co2_costs_monthly_dollars_gpt_lower = 66 * co2_total_per_month_gpt_tons
    co2_costs_monthly_dollars_gpt_upper = 200 * co2_total_per_month_gpt_tons

    #use the numbers for Gemini, 8.5 billion searches per month, 0.03 grams of carbon per average prompt
    gemini_carbon_emissions_grams = 0.03 
    ai_overviews_searches = (8.5e+9) * 30
    ai_overviews_total_emissions = (gemini_carbon_emissions_grams * (ai_overviews_searches*0.18))/(1e+6)
    co2_costs_monthly_ai_overviews_lower = ai_overviews_total_emissions * 66
    co2_costs_monthly_ai_overviews_upper = ai_overviews_total_emissions * 200

    co2_data = {
        'CO2 social costs lower bound': [co2_costs_monthly_dollars_gpt_lower,co2_costs_monthly_ai_overviews_lower],
        'CO2 social costs upper bound': [co2_costs_monthly_dollars_gpt_upper,co2_costs_monthly_ai_overviews_upper]
    }

    co2_df = pd.DataFrame(co2_data)

    #plotting the data
    labels = ['ChatGPT','Google AI Overviews (Gemini 2.5)']
    x = np.arange(len(labels))
    width = 0.35
    plt.rcParams['axes.grid'] = False

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, co2_df['CO2 social costs lower bound'], width=width, label='Lower bound ($66)', color='red', edgecolor='black')
    plt.bar(x + width/2, co2_df['CO2 social costs upper bound'], width=width, label='Upper bound ($200)', color='blue',edgecolor='black')

    plt.xlabel('Model name')
    plt.ylabel('Social cost of total emissions (in USD)')
    plt.title('Social cost of estimated total monthly carbon emissions for proprietary models')
    plt.xticks(x, labels,ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f'CO2 total per month for ChatGPT in tons: {co2_total_per_month_gpt_tons}')
    print(f'CO2 total for Google AI Overviews in tons: {ai_overviews_total_emissions}')
    #print(co2_df)



def proprietary_model_water():
    prompts_per_month_gpt = (2.5e+9) * 30
    water_per_prompt_gpt = 0.000322 
    total_per_month_water_gpt = water_per_prompt_gpt * prompts_per_month_gpt
    olive_oil_blue_water_footprint = 2388 * 1000
    corn_blue_water_footprint = 81 * 1000

    gemini_water_per_prompt = 0.34 / 1000
    ai_overviews_searches = (8.5e+9)*30
    ai_overviews_total_water = (gemini_water_per_prompt *(ai_overviews_searches*0.18))
    
    water_data = {
        'Water consumption (L)': [total_per_month_water_gpt,ai_overviews_total_water]
    }

    water_df = pd.DataFrame(water_data)

    labels = ['ChatGPT','Google AI Overviews (Gemini 2.5)']
    plt.rcParams['axes.grid'] = False
    plt.figure(figsize=(10, 6))
    plt.xlabel('Model name')
    plt.ylabel('Water consumption per month (L)')
    plt.title('Monthly water consumption of proprietary LLMs (L)')
    plt.bar(labels,water_df['Water consumption (L)'],color='blue')
    plt.show()

    print(f'ChatGPT total water per month in liters: {total_per_month_water_gpt}, in olive oil (tons): {total_per_month_water_gpt / olive_oil_blue_water_footprint}, corn: {total_per_month_water_gpt / corn_blue_water_footprint}')
    print(f'Google AI overviews total water per month in liters: {ai_overviews_total_water}')

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
    #import csv files, clean the data
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
    proprietary_model_co2()
    proprietary_model_water()