import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def fmt_val(v):
    if abs(v) < 10000:
        return f"{v:.1f}"
    return f"{v:.1e}"

def plot_training_data_co2(carbon_df: pd.DataFrame, carbon_prices_df: pd.DataFrame):
    #calculate social costs of carbon emissions using standard lower ($66) and upper ($200) SCC bounds
    carbon_df['Social cost of carbon emissions (lower bound, in USD)'] = carbon_df['CO2 (tCO2eq)'] * 66
    carbon_df['Social cost of carbon emissions (upper bound, in USD)'] = carbon_df['CO2 (tCO2eq)'] * 200

    #52.46 tCO2e per flight assumes an average load factor of 84.78% for LHR to JFK flights in 2025,
    #260 seats per flight on average, and 238 kgCO2e per passenger per flight on that route
    #according to Google Flights; i.e. 260 * 0.8478 * 238 kg = 52,461 kg ≈ 52.46 tCO2e
    lhr_jfk_flight_co2_tons = 52.46
    czech_residents_emissions_tons_per_capita = 7.04

    opp_cost_co2_emissions_flights = []
    opp_cost_co2_emissions_czech_residents = []

    for i in carbon_df['CO2 (tCO2eq)']:
        opp_cost_co2_emissions_flights.append(i / lhr_jfk_flight_co2_tons)
        opp_cost_co2_emissions_czech_residents.append(i / czech_residents_emissions_tons_per_capita)

    #plot social cost of carbon emissions
    fig, ax1 = plt.subplots(1, 1, figsize=(21, 11))
    x = np.arange(len(carbon_df['LLM model']))
    width = 0.45
    fig.subplots_adjust(bottom=0.45)

    total_lower = carbon_df['Social cost of carbon emissions (lower bound, in USD)'].sum()
    total_upper = carbon_df['Social cost of carbon emissions (upper bound, in USD)'].sum()

    bars_lower = ax1.bar(x - width/2, carbon_df['Social cost of carbon emissions (lower bound, in USD)'], width=width, label='Lower bound ($66/tCO2eq)', color='red', edgecolor='black')
    bars_upper = ax1.bar(x + width/2, carbon_df['Social cost of carbon emissions (upper bound, in USD)'], width=width, label='Upper bound ($200/tCO2eq)', color='blue', edgecolor='black')
    ax1.bar_label(bars_lower, labels=[f"${v:,.2f} ({v/total_lower*100:.1f}%)" for v in carbon_df['Social cost of carbon emissions (lower bound, in USD)']], fontsize=8, padding=2, rotation=90)
    ax1.bar_label(bars_upper, labels=[f"${v:,.2f} ({v/total_upper*100:.1f}%)" for v in carbon_df['Social cost of carbon emissions (upper bound, in USD)']], fontsize=8, padding=2, rotation=90)
    ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.set_xticks(x, carbon_df['Display Name'], rotation=45, ha='right')
    ax1.tick_params(axis='x', labelsize=7)
    ax1.set_xlabel('Model names')
    ax1.set_ylabel('Social cost of carbon emissions (USD)')
    ax1.set_title('Social cost of carbon emissions of all models in dataset (USD)')
    ax1.margins(y=0.4)
    ax1.legend()

    plt.tight_layout(pad=5.0)
    plt.show()

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(36, 8))
    fig2.subplots_adjust(bottom=0.45)
    bars4 = ax3.bar(x, opp_cost_co2_emissions_flights, width=0.9, color='blue', edgecolor='black')
    ax3.bar_label(bars4, fmt='%.3g', fontsize=6, padding=2)
    ax3.set_xlabel('Model names')
    ax3.set_xticks(x, carbon_df['Display Name'], rotation=45, ha='right')
    ax3.tick_params(axis='x', labelsize=7)
    ax3.set_ylabel('Number of one-way LHR to JFK flights [British Airways]')
    ax3.set_title('Opportunity costs of models during training\n(one-way LHR to JFK flights [British Airways])')

    bars5 = ax4.bar(x, opp_cost_co2_emissions_czech_residents, width=0.9, color='green', edgecolor='black')
    ax4.bar_label(bars5, fmt='%.3g', fontsize=6, padding=2)
    ax4.set_xlabel('Model names')
    ax4.set_ylabel('Number of Czech residents')
    ax4.set_title('Opportunity costs of training scenarios\n(equivalent Czech resident annual emissions)')
    ax4.set_xticks(x, carbon_df['Display Name'], rotation=45, ha='right')
    ax4.tick_params(axis='x', labelsize=7)

    plt.tight_layout(pad=5.0)
    plt.show()

def plot_training_data_water(water_df: pd.DataFrame, crop_prices_df: pd.DataFrame):
    #multiply the footprints from Mekonnen & Hoekstra (2011) by 1000 to convert to L/ton instead of m^3/ton
    corn_blue_water_footprint = 81 * 1000
    olive_oil_blue_water_footprint = 2388 * 1000
    bananas_blue_water_footprint = 97 * 1000
    wheat_blue_water_footprint = 342 * 1000

    #plot total water consumption of all models in the training dataset (bar chart)
    total_water = water_df['Estimated total water consumption (L)'].sum()

    fig_w, ax_w = plt.subplots(figsize=(15, 8))
    bars1 = ax_w.bar(water_df['Display Name'], water_df['Estimated total water consumption (L)'], color='blue')
    ax_w.bar_label(bars1, labels=[f"{v:,.1f} ({v/total_water*100:.1f}%)" for v in water_df['Estimated total water consumption (L)']], fontsize=6, padding=2, rotation=90)
    ax_w.set_xlabel('LLM model name')
    ax_w.set_ylabel('Total water consumption (L)')
    ax_w.set_title('Total water consumption of models during training (in L) for each model in dataset')
    ax_w.tick_params(axis='x', rotation=45)
    plt.setp(ax_w.get_xticklabels(), ha='right')
    ax_w.margins(y=0.3)
    plt.tight_layout(pad=2.0)
    plt.show()

    water_opportunity_costs_labels = ['Corn','Olive oil', 'Bananas','Wheat']
   
    total_water_consumed_training = water_df['Estimated total water consumption (L)'].sum()

    water_opportunity_costs = [
        total_water_consumed_training / corn_blue_water_footprint,
        total_water_consumed_training / olive_oil_blue_water_footprint,
        total_water_consumed_training / bananas_blue_water_footprint,
        total_water_consumed_training / wheat_blue_water_footprint
    ]

    water_opportunity_costs_monetized = [
        (total_water_consumed_training / corn_blue_water_footprint)*crop_prices_df['Yearly Average in USD per metric ton (2025)'].iloc[0],
        (total_water_consumed_training / olive_oil_blue_water_footprint)*crop_prices_df['Yearly Average in USD per metric ton (2025)'].iloc[2],
        (total_water_consumed_training / bananas_blue_water_footprint)*crop_prices_df['Yearly Average in USD per metric ton (2025)'].iloc[1],
        (total_water_consumed_training / wheat_blue_water_footprint)*crop_prices_df['Yearly Average in USD per metric ton (2025)'].iloc[3]
    ]

    #plot water opportunity costs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    bars2 = ax1.bar(water_opportunity_costs_labels,water_opportunity_costs,width=0.5,color='blue')
    ax1.bar_label(bars2, labels=[fmt_val(v) for v in water_opportunity_costs], fontsize=6, padding=2)
    ax1.set_title('Metric tons of crops that could have been grown with \n total estimated water consumed during training')
    ax1.set_xlabel('Crops')
    ax1.set_ylabel('Amount of crops (in metric tons)')

    #plot monetized value of opportunity costs
    bars3 = ax2.bar(water_opportunity_costs_labels,water_opportunity_costs_monetized,width=0.5,color='blue')
    ax2.bar_label(bars3, labels=[f"${v:,.1f}" for v in water_opportunity_costs_monetized], fontsize=6, padding=2)
    ax2.set_title('Monetized global average market cost of metric tons of crops that could have been \n grown with total estimated water consumed during training')
    ax2.set_xlabel('Crops')
    ax2.set_ylabel('Monetized global average cost of crops on market (in USD)')
    plt.show()



def clean_training_data():
    #import training data, remove unused columns, then call plotting functions
    training_df_carbon = pd.read_csv("carbon_training_data.csv")
    training_df_water = pd.read_csv("water_training_data.csv")
    crop_prices_df = pd.read_csv("global_price_of_crops.csv")
    #carbon_prices.csv uses semicolon separators and European decimal commas
    carbon_prices_df = pd.read_csv("carbon_prices.csv", sep=';', decimal=',', encoding='utf-8-sig')
    
    training_df_carbon = training_df_carbon.drop(columns=['Source',
     'Total CO2','GPU Used','Source for PUE value']
    )

    training_df_water = training_df_water.drop(
        columns=['Source',
        'Country of the organization(s) that trained the model',
        'Source for PUE value',
        'GPU Used','Source for WUE onsite value',
        'Source for WUE offsite value',
        'Sum of total water consumption (L)']
    )

    #add a Display Name column to each dataframe combining the model name and parameter count (e.g. "Llama 2-7B")
    for df in [training_df_carbon, training_df_water]:
        params = df['Parameters (billions)'].apply(lambda p: str(int(p)) + 'B' if p == int(p) else str(p) + 'B')
        df['Display Name'] = df['LLM model'] + '-' + params

    plot_training_data_co2(training_df_carbon, carbon_prices_df)
    plot_training_data_water(training_df_water, crop_prices_df)