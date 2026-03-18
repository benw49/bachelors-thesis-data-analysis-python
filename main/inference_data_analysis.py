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

    sf_ny_roundtrip_flights_co2_tons_total = 180.4
    czech_residents_emissions_tons_per_capita = 7.04

    opp_cost_co2_emissions_flights = []
    opp_cost_co2_emissions_czech_residents = []

    actual_co2_emissions = []

    for i in total_co2_emissions_lower:
        opp_cost_co2_emissions_flights.append((i/66)/sf_ny_roundtrip_flights_co2_tons_total)
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

    bars3 = ax2.bar(x,actual_co2_emissions, width=width,color='blue',edgecolor='black')
    ax2.bar_label(bars3, fmt='%.3g', fontsize=6, padding=2)
    ax2.set_ylabel('CO2 emissions (tCO2eq)')
    ax2.set_title('Total monthly CO2 emissions (tCO2eq) emissions per DAU scenario')
    ax2.set_xlabel('DAU Scenarios')
    ax2.set_xticks(x,inference_labels,rotation=45,ha='right')

    #plotting opportunity costs of carbon emissions for the DAU scenarios
    bars4 = ax3.bar(x,opp_cost_co2_emissions_flights,width=width,color='green',edgecolor='black')
    ax3.bar_label(bars4, fmt='%.3g', fontsize=6, padding=2)
    ax3.set_ylabel('Number of SFO to JFK flights')
    ax3.set_xlabel('DAU Scenarios')
    ax3.set_title('Opportunity costs of inference scenarios \n (SFO to JFK flights)')
    ax3.set_xticks(x,inference_labels,rotation=45,ha='right')

    bars5 = ax4.bar(x,opp_cost_co2_emissions_czech_residents,width=width,color='orange',edgecolor='black')
    ax4.bar_label(bars5, fmt='%.3g', fontsize=6, padding=2)
    ax4.set_ylabel('Number of czech households')
    ax4.set_xlabel('DAU Scenarios')
    ax4.set_title('Opportunity costs of inference scenarios \n (number of czech households)')
    ax4.set_xticks(x,inference_labels,rotation=45,ha='right')

    plt.tight_layout()
    plt.show()
    

def calculate_water_consumption(energy_df: pd.DataFrame, crop_prices_df: pd.DataFrame):
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

    olive_oil_blue_water_footprint = 2388 * 1000
    corn_blue_water_footprint = 81 * 1000
    wheat_blue_water_footprint = 342 * 1000
    banana_blue_water_footprint = 160 * 1000

    opp_cost_water_olive_oil = []
    opp_cost_water_corn = []
    opp_cost_water_wheat = []
    opp_cost_water_banana = []

    for i in total_water_consumption:
        opp_cost_water_olive_oil.append(i / olive_oil_blue_water_footprint)
        opp_cost_water_corn.append(i / corn_blue_water_footprint)
        opp_cost_water_wheat.append(i / wheat_blue_water_footprint)
        opp_cost_water_banana.append(i / banana_blue_water_footprint)

    #plot barchart and shift labels by 45 degrees to enhance readability, and set y-axis scale to 10^7
    #also plotting for the opportunity costs of water consumption
    plt.rcParams['axes.grid'] = False
    x = np.arange(len(inference_labels))
    width = 0.5

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(28, 6))

    bars1 = ax1.bar(x, total_water_consumption, width=width, color='blue', edgecolor='black')
    ax1.bar_label(bars1, fmt='%.3g', fontsize=6, padding=2)
    ax1.set_xticks(x, inference_labels, rotation=45, ha='right')
    ax1.set_xlabel('DAU Scenarios')
    ax1.set_ylabel('Total Water Consumption (L)')
    ax1.set_title('Total monthly water consumption (L) \n for daily active user scenarios')
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.yaxis.get_major_formatter().set_powerlimits((7, 7))

    bars2 = ax2.bar(x, opp_cost_water_olive_oil, width=width, color='green', edgecolor='black')
    ax2.bar_label(bars2, fmt='%.3g', fontsize=6, padding=2)
    ax2.set_xticks(x, inference_labels, rotation=45, ha='right')
    ax2.set_xlabel('DAU Scenarios')
    ax2.set_ylabel('Olive oil equivalent (tons)')
    ax2.set_title('Opportunity costs of inference scenarios \n (olive oil equivalent)')

    bars3 = ax3.bar(x, opp_cost_water_corn, width=width, color='orange', edgecolor='black')
    ax3.bar_label(bars3, fmt='%.3g', fontsize=6, padding=2)
    ax3.set_xticks(x, inference_labels, rotation=45, ha='right')
    ax3.set_xlabel('DAU Scenarios')
    ax3.set_ylabel('Corn equivalent (tons)')
    ax3.set_title('Opportunity costs of inference scenarios \n (corn equivalent)')

    bars4 = ax4.bar(x, opp_cost_water_wheat, width=width, color='goldenrod', edgecolor='black')
    ax4.bar_label(bars4, fmt='%.3g', fontsize=6, padding=2)
    ax4.set_xticks(x, inference_labels, rotation=45, ha='right')
    ax4.set_xlabel('DAU Scenarios')
    ax4.set_ylabel('Wheat equivalent (tons)')
    ax4.set_title('Opportunity costs of inference scenarios \n (wheat equivalent)')

    bars5 = ax5.bar(x, opp_cost_water_banana, width=width, color='yellow', edgecolor='black')
    ax5.bar_label(bars5, fmt='%.3g', fontsize=6, padding=2)
    ax5.set_xticks(x, inference_labels, rotation=45, ha='right')
    ax5.set_xlabel('DAU Scenarios')
    ax5.set_ylabel('Banana equivalent (tons)')
    ax5.set_title('Opportunity costs of inference scenarios \n (banana equivalent)')

    plt.tight_layout()
    plt.show()

    corn_price = crop_prices_df['Yearly Average in USD per metric ton (2025)'].iloc[0]
    banana_price = crop_prices_df["Yearly Average in USD per metric ton (2025)"].iloc[1]
    olive_oil_price = crop_prices_df["Yearly Average in USD per metric ton (2025)"].iloc[2]
    wheat_price = crop_prices_df['Yearly Average in USD per metric ton (2025)'].iloc[3]

    econ_value_olive_oil = []
    econ_value_corn = []
    econ_value_wheat = []
    econ_value_banana = []

    for i in opp_cost_water_olive_oil:
        econ_value_olive_oil.append(i * olive_oil_price)

    for i in opp_cost_water_corn:
        econ_value_corn.append(i * corn_price)

    for i in opp_cost_water_wheat:
        econ_value_wheat.append(i * wheat_price)

    for i in opp_cost_water_banana:
        econ_value_banana.append(i * banana_price)

    fig2, (ax6, ax7, ax8, ax9) = plt.subplots(1, 4, figsize=(28, 6))

    bars6 = ax6.bar(x, econ_value_olive_oil, width=width, color='green', edgecolor='black')
    ax6.bar_label(bars6, fmt='%.3g', fontsize=6, padding=2)
    ax6.set_xticks(x, inference_labels, rotation=45, ha='right')
    ax6.set_xlabel('DAU Scenarios')
    ax6.set_ylabel('Economic value (USD)')
    ax6.set_title('Economic value of opportunity costs \n of water consumption\n(olive oil equivalent, inference scenarios)')

    bars7 = ax7.bar(x, econ_value_corn, width=width, color='orange', edgecolor='black')
    ax7.bar_label(bars7, fmt='%.3g', fontsize=6, padding=2)
    ax7.set_xticks(x, inference_labels, rotation=45, ha='right')
    ax7.set_xlabel('DAU Scenarios')
    ax7.set_ylabel('Economic value (USD)')
    ax7.set_title('Economic value of opportunity costs \n of water consumption\n(corn equivalent, inference scenarios)')

    bars8 = ax8.bar(x, econ_value_wheat, width=width, color='goldenrod', edgecolor='black')
    ax8.bar_label(bars8, fmt='%.3g', fontsize=6, padding=2)
    ax8.set_xticks(x, inference_labels, rotation=45, ha='right')
    ax8.set_xlabel('DAU Scenarios')
    ax8.set_ylabel('Economic value (USD)')
    ax8.set_title('Economic value of opportunity costs \n of water consumption\n(wheat equivalent, inference scenarios)')

    bars9 = ax9.bar(x, econ_value_banana, width=width, color='yellow', edgecolor='black')
    ax9.bar_label(bars9, fmt='%.3g', fontsize=6, padding=2)
    ax9.set_xticks(x, inference_labels, rotation=45, ha='right')
    ax9.set_xlabel('DAU Scenarios')
    ax9.set_ylabel('Economic value (USD)')
    ax9.set_title('Economic value of opportunity costs \n of water consumption\n(banana equivalent, inference scenarios)')

    plt.tight_layout(pad=5.0)
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
    ai_overviews_searches = (13.7e+9) * 30
    ai_overviews_total_emissions = (gemini_carbon_emissions_grams * (ai_overviews_searches*0.18))/(1e+6)
    co2_costs_monthly_ai_overviews_lower = ai_overviews_total_emissions * 66
    co2_costs_monthly_ai_overviews_upper = ai_overviews_total_emissions * 200

    sf_ny_roundtrip_flights_co2_tons_total = 180.4
    czech_residents_emissions_tons_per_capita = 7.04

    co2_data = {
        'CO2 social costs lower bound': [co2_costs_monthly_dollars_gpt_lower,co2_costs_monthly_ai_overviews_lower],
        'CO2 social costs upper bound': [co2_costs_monthly_dollars_gpt_upper,co2_costs_monthly_ai_overviews_upper],
        'CO2 opp cost roundtrip SFO to JFK flights': [(co2_total_per_month_gpt_tons/sf_ny_roundtrip_flights_co2_tons_total),
        (ai_overviews_total_emissions/sf_ny_roundtrip_flights_co2_tons_total)],
        'CO2 opp cost czech residents': [(co2_total_per_month_gpt_tons/czech_residents_emissions_tons_per_capita),
        (ai_overviews_total_emissions/czech_residents_emissions_tons_per_capita)]
    }

    co2_df = pd.DataFrame(co2_data)

    #plotting the data
    labels = ['ChatGPT, Mid 2025','Google AI Overviews (Gemini 2.5), Early 2025']
    x = np.arange(len(labels))
    width = 0.35
    plt.rcParams['axes.grid'] = False

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    bars1 = ax1.bar(x - width/2, co2_df['CO2 social costs lower bound'], width=width, label='Lower bound ($66)', color='red', edgecolor='black')
    ax1.bar_label(bars1,fmt='%.3g', fontsize=6, padding=2)
    bars2 = ax1.bar(x + width/2, co2_df['CO2 social costs upper bound'], width=width, label='Upper bound ($200)', color='blue',edgecolor='black')
    ax1.bar_label(bars2,fmt='%.3g', fontsize=6, padding=2)
   
    ax1.set_xlabel('Model name')
    ax1.set_ylabel('Social cost of total emissions (in USD)')
    ax1.set_title('Social cost of estimated total monthly \n carbon emissions for proprietary models')
    ax1.set_xticks(x,labels,ha='right',rotation=10)
    ax1.legend()

    bars3 = ax2.bar(x,co2_df['CO2 opp cost roundtrip SFO to JFK flights'],width=width,color='green',edgecolor='black')
    ax2.bar_label(bars3, fmt='%.3g', fontsize=6, padding=2)
    ax2.set_xlabel('Model name')
    ax2.set_ylabel('Number of roundtrip SFO to JFK flights')
    ax2.set_xticks(x,labels,ha='right',rotation=10)
    ax2.set_title('Opportunity cost of proprietary models CO2 emissions \n roundtrip (SFO to JFK flights)')

    bars4 = ax3.bar(x,co2_df['CO2 opp cost czech residents'],width=width,color='orange',edgecolor='black')
    ax3.bar_label(bars4, fmt='%.3g', fontsize=6, padding=2)
    ax3.set_xlabel('Model name')
    ax3.set_ylabel('Number of czech residents')
    ax3.set_title('Opportunity costs of proprietary models CO2 emissions \n (equivalent Czech resident annual emissions)')
    ax3.set_xticks(x,labels,ha='right',rotation=10)

    plt.tight_layout(pad=2.0, w_pad=6.0)
    plt.subplots_adjust(left=0.07)
    plt.show()

    print(f'CO2 total per month for ChatGPT in tons: {co2_total_per_month_gpt_tons}')
    print(f'CO2 total for Google AI Overviews in tons: {ai_overviews_total_emissions}')

def proprietary_model_water(crop_prices_df: pd.DataFrame):
    prompts_per_month_gpt = (2.5e+9) * 30
    water_per_prompt_gpt = 0.000322
    total_per_month_water_gpt = water_per_prompt_gpt * prompts_per_month_gpt
    olive_oil_blue_water_footprint = 2388 * 1000
    corn_blue_water_footprint = 81 * 1000
    wheat_blue_water_footprint = 342 * 1000
    banana_blue_water_footprint = 160 * 1000

    gemini_water_per_prompt = 0.34 / 1000
    ai_overviews_searches = (13.7e+9)*30
    ai_overviews_total_water = (gemini_water_per_prompt *(ai_overviews_searches*0.18))

    water_data = {
        'Water consumption (L)': [total_per_month_water_gpt, ai_overviews_total_water],
        'Water opp cost olive oil (tons)': [total_per_month_water_gpt / olive_oil_blue_water_footprint, ai_overviews_total_water / olive_oil_blue_water_footprint],
        'Water opp cost corn (tons)': [total_per_month_water_gpt / corn_blue_water_footprint, ai_overviews_total_water / corn_blue_water_footprint],
        'Water opp cost wheat (tons)': [total_per_month_water_gpt / wheat_blue_water_footprint, ai_overviews_total_water / wheat_blue_water_footprint],
        'Water opp cost banana (tons)': [total_per_month_water_gpt / banana_blue_water_footprint, ai_overviews_total_water / banana_blue_water_footprint],
    }

    water_df = pd.DataFrame(water_data)

    #plot water consumption and opportunity costs
    labels = ['ChatGPT, Mid 2025','Google AI Overviews (Gemini 2.5), Early 2025']
    x = np.arange(len(labels))
    width = 0.5
    plt.rcParams['axes.grid'] = False

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(28, 6))

    bars1 = ax1.bar(x, water_df['Water consumption (L)'], width=width, color='blue', edgecolor='black')
    ax1.bar_label(bars1, fmt='%.3g', fontsize=6, padding=2)
    ax1.set_xlabel('Model name', fontsize=8)
    ax1.set_ylabel('Water consumption per month (L)', fontsize=8)
    ax1.set_title('Monthly water consumption \n of proprietary LLMs (L)', fontsize=8)
    ax1.set_xticks(x, labels, ha='right', rotation=45, fontsize=7)
    ax1.tick_params(axis='y', labelsize=7)

    bars2 = ax2.bar(x, water_df['Water opp cost olive oil (tons)'], width=width, color='green', edgecolor='black')
    ax2.bar_label(bars2, fmt='%.3g', fontsize=6, padding=2)
    ax2.set_xlabel('Model name', fontsize=8)
    ax2.set_ylabel('Olive oil equivalent (tons)', fontsize=8)
    ax2.set_title('Opportunity costs of proprietary models \n water consumption (olive oil equivalent)', fontsize=8)
    ax2.set_xticks(x, labels, ha='right', rotation=45, fontsize=7)
    ax2.tick_params(axis='y', labelsize=7)

    bars3 = ax3.bar(x, water_df['Water opp cost corn (tons)'], width=width, color='orange', edgecolor='black')
    ax3.bar_label(bars3, fmt='%.3g', fontsize=6, padding=2)
    ax3.set_xlabel('Model name', fontsize=8)
    ax3.set_ylabel('Corn equivalent (tons)', fontsize=8)
    ax3.set_title('Opportunity costs of proprietary models \n water consumption (corn equivalent)', fontsize=8)
    ax3.set_xticks(x, labels, ha='right', rotation=45, fontsize=7)
    ax3.tick_params(axis='y', labelsize=7)

    bars4 = ax4.bar(x, water_df['Water opp cost wheat (tons)'], width=width, color='goldenrod', edgecolor='black')
    ax4.bar_label(bars4, fmt='%.3g', fontsize=6, padding=2)
    ax4.set_xlabel('Model name', fontsize=8)
    ax4.set_ylabel('Wheat equivalent (tons)', fontsize=8)
    ax4.set_title('Opportunity costs of proprietary models \n water consumption (wheat equivalent)', fontsize=8)
    ax4.set_xticks(x, labels, ha='right', rotation=45, fontsize=7)
    ax4.tick_params(axis='y', labelsize=7)

    bars5 = ax5.bar(x, water_df['Water opp cost banana (tons)'], width=width, color='yellow', edgecolor='black')
    ax5.bar_label(bars5, fmt='%.3g', fontsize=6, padding=2)
    ax5.set_xlabel('Model name', fontsize=8)
    ax5.set_ylabel('Banana equivalent (tons)', fontsize=8)
    ax5.set_title('Opportunity costs of proprietary models \n water consumption (banana equivalent)', fontsize=8)
    ax5.set_xticks(x, labels, ha='right', rotation=45, fontsize=7)
    ax5.tick_params(axis='y', labelsize=7)

    plt.tight_layout(pad=1.0, w_pad=3.0)
    plt.subplots_adjust(left=0.03)
    plt.show()

    corn_price = crop_prices_df['Yearly Average in USD per metric ton (2025)'].iloc[0]
    banana_price = crop_prices_df["Yearly Average in USD per metric ton (2025)"].iloc[1]
    olive_oil_price = crop_prices_df["Yearly Average in USD per metric ton (2025)"].iloc[2]
    wheat_price = crop_prices_df['Yearly Average in USD per metric ton (2025)'].iloc[3]

    econ_value_olive_oil = []
    econ_value_corn = []
    econ_value_wheat = []
    econ_value_banana = []

    for v in water_df['Water opp cost olive oil (tons)']:
        econ_value_olive_oil.append(v * olive_oil_price)

    for v in water_df['Water opp cost corn (tons)']:
        econ_value_corn.append(v * corn_price)

    for v in water_df['Water opp cost wheat (tons)']:
        econ_value_wheat.append(v * wheat_price)

    for v in water_df['Water opp cost banana (tons)']:
        econ_value_banana.append(v * banana_price)

    fig2, (ax6, ax7, ax8, ax9) = plt.subplots(1, 4, figsize=(28, 6))

    bars6 = ax6.bar(x, econ_value_olive_oil, width=width, color='green', edgecolor='black')
    ax6.bar_label(bars6, fmt='%.3g', fontsize=6, padding=2)
    ax6.set_xlabel('Model name', fontsize=8)
    ax6.set_ylabel('Economic value (USD)', fontsize=8)
    ax6.set_title('Economic value of opportunity costs of \n water consumption (olive oil equivalent, \n proprietary models)', fontsize=8)
    ax6.set_xticks(x, labels, ha='right', rotation=45, fontsize=7)
    ax6.tick_params(axis='y', labelsize=7)

    bars7 = ax7.bar(x, econ_value_corn, width=width, color='orange', edgecolor='black')
    ax7.bar_label(bars7, fmt='%.3g', fontsize=6, padding=2)
    ax7.set_xlabel('Model name', fontsize=8)
    ax7.set_ylabel('Economic value (USD)', fontsize=8)
    ax7.set_title('Economic value of opportunity costs of \n water consumption (corn equivalent, \n proprietary models)', fontsize=8)
    ax7.set_xticks(x, labels, ha='right', rotation=45, fontsize=7)
    ax7.tick_params(axis='y', labelsize=7)

    bars8 = ax8.bar(x, econ_value_wheat, width=width, color='goldenrod', edgecolor='black')
    ax8.bar_label(bars8, fmt='%.3g', fontsize=6, padding=2)
    ax8.set_xlabel('Model name', fontsize=8)
    ax8.set_ylabel('Economic value (USD)', fontsize=8)
    ax8.set_title('Economic value of opportunity costs of \n water consumption (wheat equivalent, \n proprietary models)', fontsize=8)
    ax8.set_xticks(x, labels, ha='right', rotation=45, fontsize=7)
    ax8.tick_params(axis='y', labelsize=7)

    bars9 = ax9.bar(x, econ_value_banana, width=width, color='yellow', edgecolor='black')
    ax9.bar_label(bars9, fmt='%.3g', fontsize=6, padding=2)
    ax9.set_xlabel('Model name', fontsize=8)
    ax9.set_ylabel('Economic value (USD)', fontsize=8)
    ax9.set_title('Economic value of opportunity costs of \n water consumption (banana equivalent, \n proprietary models)', fontsize=8)
    ax9.set_xticks(x, labels, ha='right', rotation=45, fontsize=7)
    ax9.tick_params(axis='y', labelsize=7)

    plt.tight_layout(pad=2.0, w_pad=5.0)
    plt.subplots_adjust(left=0.06)
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
    crop_prices_df = pd.read_csv("global_price_of_crops.csv")

    downloads_data.rename(columns={"Model":"fullname"}, inplace=True)
    training_data.rename(columns={"Hugging Face Model Name":"fullname"}, inplace=True)
    
    co2_df = find_alike_models(["fullname", "CO2 cost (kg)"], downloads_data, leaderboard_data)

    #create a new column in the main (cleaned) dataset 
    #this represents the average co2 cost per prompt for hugging face's analysis, since they ran benchmarks
    #that totaled 21,682 prompts of varying lengths and complexities in their openLLM leaderboard
    co2_df['Average CO2 cost (kg) per prompt'] = co2_df['CO2 cost (kg)'] / 21682
    co2_df['Average Energy Cost (kWh) per prompt'] = (co2_df['CO2 cost (kg)'] * 1000)/(269.8*21682)

    calculate_co2_costs(co2_df,training_data,["fullname", "CO2 cost (kg)"])
    calculate_water_consumption(co2_df, crop_prices_df)
    proprietary_model_co2()
    proprietary_model_water(crop_prices_df)