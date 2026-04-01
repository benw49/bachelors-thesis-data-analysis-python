import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

def fmt_dollar(v):
    return f"${v:,.2f}"

def fmt_val(v):
    #avoid scientific notation for values under 100,000
    if abs(v) < 100_000:
        if abs(v) < 1:
            return f"{v:.3f}"
        return f"{v:,.1f}"
    #use comma-separated integer format for large values to avoid scientific notation
    return f"{v:,.0f}"

def calculate_co2_costs(co2_df: pd.DataFrame, training_df: pd.DataFrame, cols):
    #calculate total inference emissions for a scenario where there are different amounts of inferences
    # *assuming that an average user does 5, 25, or 50 prompts per day (150, 750, or 1,500 per month)*
    # *assuming also that said user has a Hugging Face style setup, i.e. they use 8 H100 GPUs for inference and are located
    # in Virginia*
    # *and assuming different ranges, that 15-40% of downloaders are Daily Active Users (DAU), since not every download may be an active user*
    #then convert total emissions from kg to tons
    DAU = [0.15,0.2,0.3,0.4]
    inferences = [150,750,1500]
    inference_labels = get_inference_labels(inferences)

    total_co2_emissions_lower = []
    total_co2_emissions_upper = []

    for user_scaler in DAU:
        for I in inferences: 
            co2_emissions_users_lower = ((co2_df['Average CO2 cost (kg) per prompt'] * I * user_scaler * co2_df['downloads']) / 1000).sum() * 66
            co2_emissions_users_upper = ((co2_df['Average CO2 cost (kg) per prompt'] * I * user_scaler * co2_df['downloads']) / 1000).sum() * 200
            total_co2_emissions_lower.append(co2_emissions_users_lower)
            total_co2_emissions_upper.append(co2_emissions_users_upper)

    #52.46 tCO2e per flight assumes an average load factor of 84.78% for LHR to JFK flights in 2025,
    #260 seats per flight on average, and 238 kgCO2e per passenger per flight on that route
    #according to Google Flights; i.e. 260 * 0.8478 * 238 kg = 52,461 kg ≈ 52.46 tCO2e
    lhr_jfk_flight_co2_tons = 52.46
    czech_residents_emissions_tons_per_capita = 7.04

    opp_cost_co2_emissions_flights = []
    opp_cost_co2_emissions_czech_residents = []

    for i in total_co2_emissions_lower:
        opp_cost_co2_emissions_flights.append((i/66)/lhr_jfk_flight_co2_tons)
        opp_cost_co2_emissions_czech_residents.append((i/66)/czech_residents_emissions_tons_per_capita)

    #plot social cost of estimated monthly co2 emissions in its own figure
    #labels show full dollar amount below $1M, scientific notation at or above $1M
    plt.rcParams['axes.grid'] = False
    x = np.arange(len(inference_labels))
    width = 0.35

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars_lower = ax1.bar(x - width/2, total_co2_emissions_lower, width=width, label='Lower bound ($66)', color='red', edgecolor='black')
    ax1.bar_label(bars_lower, labels=[fmt_dollar(v) for v in total_co2_emissions_lower], fontsize=6, padding=2, rotation=90)
    bars_upper = ax1.bar(x + width/2, total_co2_emissions_upper, width=width, label='Upper bound ($200)', color='blue', edgecolor='black')
    ax1.bar_label(bars_upper, labels=[fmt_dollar(v) for v in total_co2_emissions_upper], fontsize=6, padding=2, rotation=90)
    ax1.set_xticks(x, inference_labels, rotation=45, ha='right')
    ax1.set_xlabel('DAU/MAU Scenarios')
    ax1.set_ylabel('Social cost of total emissions (in USD)')
    ax1.set_title('Social cost of estimated total monthly carbon emissions\nfor DAU/MAU scenarios')
    ax1.legend()
    ax1.margins(y=0.25)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    plt.tight_layout()
    plt.show()

    #plot opportunity costs of co2 emissions for the DAU scenarios in one figure with two subplots
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 6))

    bars4 = ax2.bar(x, opp_cost_co2_emissions_flights, width=width, color='green', edgecolor='black')
    ax2.bar_label(bars4, labels=[fmt_val(v) for v in opp_cost_co2_emissions_flights], fontsize=6, padding=2)
    ax2.set_ylabel('Number of one-way LHR to JFK flights [British Airways]')
    ax2.set_xlabel('DAU/MAU Scenarios')
    ax2.set_title('Opportunity costs of inference scenarios\n(one-way LHR to JFK flights [British Airways])')
    ax2.set_xticks(x, inference_labels, rotation=45, ha='right')

    bars5 = ax3.bar(x, opp_cost_co2_emissions_czech_residents, width=width, color='orange', edgecolor='black')
    ax3.bar_label(bars5, labels=[fmt_val(v) for v in opp_cost_co2_emissions_czech_residents], fontsize=6, padding=2)
    ax3.set_ylabel('Number of Czech residents')
    ax3.set_xlabel('DAU/MAU Scenarios')
    ax3.set_title('Opportunity costs of inference scenarios\n(equivalent Czech resident annual emissions)')
    ax3.set_xticks(x, inference_labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
    

def calculate_water_consumption(energy_df: pd.DataFrame, crop_prices_df: pd.DataFrame):
    #WUE and PUE values assuming HuggingFace likely used AWS us-east-1 servers in Northern Virginia
    onsiteWUE = 0.12
    offsiteWUE = 0.63
    PUE = 1.15

    energy_df['Average water consumption (L) per prompt'] = energy_df['Average Energy Cost (kWh) per prompt'] * (onsiteWUE+(PUE*offsiteWUE))

    #5, 25, or 50 prompts per day multiplied by 30 days to get monthly inferences
    inferences = [150,750,1500]
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

    #plot water consumption as a standalone bar chart
    plt.rcParams['axes.grid'] = False
    x = np.arange(len(inference_labels))
    width = 0.5

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars1 = ax1.bar(x, total_water_consumption, width=width, color='blue', edgecolor='black')
    ax1.bar_label(bars1, labels=[fmt_val(v) for v in total_water_consumption], fontsize=6, padding=2)
    ax1.set_xticks(x, inference_labels, rotation=45, ha='right')
    ax1.set_xlabel('DAU/MAU Scenarios')
    ax1.set_ylabel('Total Water Consumption (L)')
    ax1.set_title('Total monthly water consumption (L) \n for daily active user scenarios')
    #use comma-separated integer format on the y-axis to suppress scientific notation
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
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

    #plot opportunity costs (top row) and monetized opportunity costs (bottom row) in one combined figure
    opp_data = [opp_cost_water_olive_oil, opp_cost_water_corn, opp_cost_water_wheat, opp_cost_water_banana]
    econ_data = [econ_value_olive_oil, econ_value_corn, econ_value_wheat, econ_value_banana]
    opp_labels = ['Olive oil equivalent (tons)', 'Corn equivalent (tons)', 'Wheat equivalent (tons)', 'Banana equivalent (tons)']
    econ_labels = ['Olive oil equivalent (USD)', 'Corn equivalent (USD)', 'Wheat equivalent (USD)', 'Banana equivalent (USD)']
    crop_titles = ['Olive Oil', 'Corn', 'Wheat', 'Banana']
    opp_colors = ['green', 'orange', 'goldenrod', 'yellow']

    fig2, axes = plt.subplots(2, 4, figsize=(28, 12))

    for col, (opp_vals, econ_vals, opp_ylabel, econ_ylabel, title, color) in enumerate(
        zip(opp_data, econ_data, opp_labels, econ_labels, crop_titles, opp_colors)
    ):
        ax_top = axes[0, col]
        bars = ax_top.bar(x, opp_vals, width=width, color=color, edgecolor='black')
        ax_top.bar_label(bars, labels=[fmt_val(v) for v in opp_vals], fontsize=6, padding=2)
        ax_top.set_xticks(x, inference_labels, rotation=45, ha='right')
        ax_top.set_xlabel('DAU/MAU Scenarios')
        ax_top.set_ylabel(opp_ylabel)
        ax_top.set_title(f'Opportunity costs of inference scenarios\n({title} equivalent)')
        ax_top.margins(y=0.2)

        ax_bot = axes[1, col]
        bars = ax_bot.bar(x, econ_vals, width=width, color=color, edgecolor='black')
        ax_bot.bar_label(bars, labels=[fmt_val(v) for v in econ_vals], fontsize=6, padding=2)
        ax_bot.set_xticks(x, inference_labels, rotation=45, ha='right')
        ax_bot.set_xlabel('DAU/MAU Scenarios')
        ax_bot.set_ylabel(econ_ylabel)
        ax_bot.set_title(f'Monetized opportunity costs of water consumption\n({title} equivalent, inference scenarios)')
        ax_bot.margins(y=0.2)

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

    #use the numbers for Gemini, 13.7 billion searches per month, 0.03 grams of carbon per average prompt
    gemini_carbon_emissions_grams = 0.03 
    ai_overviews_searches = (13.7e+9) * 30
    ai_overviews_total_emissions = (gemini_carbon_emissions_grams * (ai_overviews_searches*0.18))/(1e+6)
    co2_costs_monthly_ai_overviews_lower = ai_overviews_total_emissions * 66
    co2_costs_monthly_ai_overviews_upper = ai_overviews_total_emissions * 200

    #52.46 tCO2e per flight assumes an average load factor of 84.78% for LHR to JFK flights in 2025,
    #260 seats per flight on average, and 238 kgCO2e per passenger per flight on that route
    #according to Google Flights; i.e. 260 * 0.8478 * 238 kg = 52,461 kg ≈ 52.46 tCO2e
    lhr_jfk_flight_co2_tons = 52.46
    czech_residents_emissions_tons_per_capita = 7.04

    co2_data = {
        'CO2 social costs lower bound': [co2_costs_monthly_dollars_gpt_lower,co2_costs_monthly_ai_overviews_lower],
        'CO2 social costs upper bound': [co2_costs_monthly_dollars_gpt_upper,co2_costs_monthly_ai_overviews_upper],
        'CO2 opp cost one-way LHR to JFK flights (British Airways)': [(co2_total_per_month_gpt_tons/lhr_jfk_flight_co2_tons),
        (ai_overviews_total_emissions/lhr_jfk_flight_co2_tons)],
        'CO2 opp cost czech residents': [(co2_total_per_month_gpt_tons/czech_residents_emissions_tons_per_capita),
        (ai_overviews_total_emissions/czech_residents_emissions_tons_per_capita)]
    }

    co2_df = pd.DataFrame(co2_data)

    #plot social cost of estimated total monthly carbon emissions for proprietary models in its own figure
    #labels show full dollar amount below $1M, scientific notation at or above $1M
    labels = ['ChatGPT, Mid 2025','Google AI Overviews (Gemini), Early 2025']
    x = np.arange(len(labels))
    width = 0.35
    plt.rcParams['axes.grid'] = False

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    bars1 = ax1.bar(x - width/2, co2_df['CO2 social costs lower bound'], width=width, label='Lower bound ($66)', color='red', edgecolor='black')
    ax1.bar_label(bars1, labels=[fmt_dollar(v) for v in co2_df['CO2 social costs lower bound']], fontsize=6, padding=2)
    bars2 = ax1.bar(x + width/2, co2_df['CO2 social costs upper bound'], width=width, label='Upper bound ($200)', color='blue', edgecolor='black')
    ax1.bar_label(bars2, labels=[fmt_dollar(v) for v in co2_df['CO2 social costs upper bound']], fontsize=6, padding=2)
    ax1.set_xlabel('Model name')
    ax1.set_ylabel('Social cost of total emissions (in USD)')
    ax1.set_title('Social cost of estimated total monthly\ncarbon emissions for proprietary models')
    ax1.set_xticks(x, labels, ha='right', rotation=10)
    ax1.legend()
    plt.tight_layout()
    plt.show()

    #plot opportunity costs of proprietary models CO2 emissions in one figure with two subplots
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 6))

    bars3 = ax2.bar(x, co2_df['CO2 opp cost one-way LHR to JFK flights (British Airways)'], width=width, color='green', edgecolor='black')
    ax2.bar_label(bars3, labels=[fmt_val(v) for v in co2_df['CO2 opp cost one-way LHR to JFK flights (British Airways)']], fontsize=6, padding=2)
    ax2.set_xlabel('Model name')
    ax2.set_ylabel('Number of one-way LHR to JFK flights (British Airways)')
    ax2.set_xticks(x, labels, ha='right', rotation=10)
    ax2.set_title('Opportunity cost of proprietary models CO2 emissions\n(one-way LHR to JFK flights (British Airways))')

    bars4 = ax3.bar(x, co2_df['CO2 opp cost czech residents'], width=width, color='orange', edgecolor='black')
    ax3.bar_label(bars4, labels=[fmt_val(v) for v in co2_df['CO2 opp cost czech residents']], fontsize=6, padding=2)
    ax3.set_xlabel('Model name')
    ax3.set_ylabel('Number of Czech residents')
    ax3.set_title('Opportunity costs of proprietary models CO2 emissions\n(equivalent Czech resident annual emissions)')
    ax3.set_xticks(x, labels, ha='right', rotation=10)

    plt.tight_layout()
    plt.show()

def proprietary_model_water(crop_prices_df: pd.DataFrame):
    prompts_per_month_gpt = (2.5e+9) * 30
    water_per_prompt_gpt = 0.000322
    total_per_month_water_gpt = water_per_prompt_gpt * prompts_per_month_gpt
    olive_oil_blue_water_footprint = 2388 * 1000
    corn_blue_water_footprint = 81 * 1000
    wheat_blue_water_footprint = 342 * 1000
    banana_blue_water_footprint = 97 * 1000

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

    #plot monthly water consumption in its own standalone bar chart
    labels = ['ChatGPT, Mid 2025','Google AI Overviews (Gemini), Early 2025']
    x = np.arange(len(labels))
    width = 0.5
    plt.rcParams['axes.grid'] = False

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    bars1 = ax1.bar(x, water_df['Water consumption (L)'], width=width, color='blue', edgecolor='black')
    ax1.bar_label(bars1, labels=[f"{v:,.1f}" for v in water_df['Water consumption (L)']], fontsize=8, padding=2)
    ax1.set_xlabel('Model name')
    ax1.set_ylabel('Water consumption per month (L)')
    ax1.set_title('Monthly water consumption of proprietary LLMs (L)')
    ax1.set_xticks(x, labels, ha='right', rotation=10)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
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

    for v in water_df['Water opp cost olive oil (tons)']:
        econ_value_olive_oil.append(v * olive_oil_price)

    for v in water_df['Water opp cost corn (tons)']:
        econ_value_corn.append(v * corn_price)

    for v in water_df['Water opp cost wheat (tons)']:
        econ_value_wheat.append(v * wheat_price)

    for v in water_df['Water opp cost banana (tons)']:
        econ_value_banana.append(v * banana_price)

    #group data by crop, with one bar per model within each crop group
    crops = ['Olive Oil', 'Corn', 'Wheat', 'Banana']
    model_colors = ['steelblue', 'tomato']
    x_crops = np.arange(len(crops))
    width = 0.35

    opp_costs_gpt = [
        water_df['Water opp cost olive oil (tons)'].iloc[0],
        water_df['Water opp cost corn (tons)'].iloc[0],
        water_df['Water opp cost wheat (tons)'].iloc[0],
        water_df['Water opp cost banana (tons)'].iloc[0],
    ]
    opp_costs_gemini = [
        water_df['Water opp cost olive oil (tons)'].iloc[1],
        water_df['Water opp cost corn (tons)'].iloc[1],
        water_df['Water opp cost wheat (tons)'].iloc[1],
        water_df['Water opp cost banana (tons)'].iloc[1],
    ]
    econ_gpt = [econ_value_olive_oil[0], econ_value_corn[0], econ_value_wheat[0], econ_value_banana[0]]
    econ_gemini = [econ_value_olive_oil[1], econ_value_corn[1], econ_value_wheat[1], econ_value_banana[1]]

    fig2, (ax_opp, ax_econ) = plt.subplots(1, 2, figsize=(16, 6))

    bars_opp_gpt = ax_opp.bar(x_crops - width/2, opp_costs_gpt, width=width, label='ChatGPT, Mid 2025', color=model_colors[0], edgecolor='black')
    ax_opp.bar_label(bars_opp_gpt, labels=[f"{v:,.1f}" for v in opp_costs_gpt], fontsize=7, padding=2)
    bars_opp_gemini = ax_opp.bar(x_crops + width/2, opp_costs_gemini, width=width, label='Google AI Overviews (Gemini), Early 2025', color=model_colors[1], edgecolor='black')
    ax_opp.bar_label(bars_opp_gemini, labels=[f"{v:,.1f}" for v in opp_costs_gemini], fontsize=7, padding=2)
    ax_opp.set_xlabel('Crop')
    ax_opp.set_ylabel('Opportunity cost (metric tons)')
    ax_opp.set_title('Opportunity costs of proprietary model\nwater consumption by crop (metric tons)')
    ax_opp.set_xticks(x_crops, crops)
    ax_opp.legend()
    ax_opp.margins(y=0.2)
    ax_opp.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    bars_econ_gpt = ax_econ.bar(x_crops - width/2, econ_gpt, width=width, label='ChatGPT, Mid 2025', color=model_colors[0], edgecolor='black')
    ax_econ.bar_label(bars_econ_gpt, labels=[fmt_dollar(v) for v in econ_gpt], fontsize=7, padding=2)
    bars_econ_gemini = ax_econ.bar(x_crops + width/2, econ_gemini, width=width, label='Google AI Overviews (Gemini), Early 2025', color=model_colors[1], edgecolor='black')
    ax_econ.bar_label(bars_econ_gemini, labels=[fmt_dollar(v) for v in econ_gemini], fontsize=7, padding=2)
    ax_econ.set_xlabel('Crop')
    ax_econ.set_ylabel('Economic value (USD)')
    ax_econ.set_title('Monetized opportunity costs of proprietary model\nwater consumption by crop (USD)')
    ax_econ.set_xticks(x_crops, crops)
    ax_econ.legend()
    ax_econ.margins(y=0.2)
    ax_econ.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    plt.tight_layout()
    plt.show()

#function for finding out which of the models form the datasets can be found in the downloads data
def find_alike_models(cols, df1: pd.DataFrame, df2: pd.DataFrame):
    mask = df2['fullname'].isin(df1['fullname'])
    df_filtered = df2.loc[mask,cols]
    df_filtered_final = pd.merge(df_filtered,df1,on="fullname")
    return df_filtered_final

#function that gets the inference labels in the proper order for graphing on a bar chart
def get_inference_labels(inferences):
    scenarios_labels = ['15%','20%','30%','40%']
    inference_categories = ['5 prompts/day', '25 prompts/day', '50 prompts/day']

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