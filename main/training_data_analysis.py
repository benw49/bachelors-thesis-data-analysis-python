import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


#rounds dollar values 
#values over $1M are shown as e.g. "$2.7M" for readability
def fmt_money_rounded(v):
    if abs(v) >= 1_000_000:
        return f"${v/1e6:.1f}M"
    elif abs(v) >= 10_000:
        return f"${round(v, -3)/1000:.0f}K"
    elif abs(v) >= 1_000:
        return f"${round(v, -2)/1000:.1f}K"
    else:
        return f"${v:,.0f}"


#same rounding logic as fmt_money_rounded but for non-dollar counts (liters, metric tons, etc.)
def fmt_count_rounded(v):
    if abs(v) >= 1_000_000:
        return f"{v/1e6:.1f}M"
    elif abs(v) >= 10_000:
        return f"{round(v, -3)/1000:.0f}K"
    elif abs(v) >= 1_000:
        return f"{round(v, -2)/1000:.1f}K"
    else:
        return f"{v:,.0f}"


def plot_training_data_co2(carbon_df: pd.DataFrame):
    carbon_df['Social cost of carbon emissions (lower bound, in USD)'] = carbon_df['CO2 (tCO2eq)'] * 66
    carbon_df['Social cost of carbon emissions (upper bound, in USD)'] = carbon_df['CO2 (tCO2eq)'] * 200

    #66,269 kgCO2 total for 219 passengers (economy class) on a Heathrow to JFK flight,
    #as determined using the ICAO emissions calculator; i.e. 66,269 kg = 66.269 tCO2 per flight
    lhr_jfk_flight_co2_tons = 66.269
    czech_residents_emissions_tons_per_capita = 7.04

    # sort descending by lower bound, split top 10 vs remaining
    carbon_sorted = carbon_df.sort_values(
        'Social cost of carbon emissions (lower bound, in USD)', ascending=False
    ).reset_index(drop=True)
    top10 = carbon_sorted.iloc[:10].reset_index(drop=True)
    rest = carbon_sorted.iloc[10:].reset_index(drop=True)

    opp_cost_co2_emissions_flights = []
    opp_cost_co2_emissions_czech_residents = []

    for i in carbon_sorted['CO2 (tCO2eq)']:
        opp_cost_co2_emissions_flights.append(i / lhr_jfk_flight_co2_tons)
        opp_cost_co2_emissions_czech_residents.append(i / czech_residents_emissions_tons_per_capita)

    width = 0.35
    #Graph: Social cost of carbon emissions — Top 10 models (upper subplot) and remaining models (lower subplot)
    fig, (ax_top, ax_rest) = plt.subplots(2, 1, figsize=(20, 22))

    x_top = np.arange(len(top10))
    bars_lower_top = ax_top.bar(
        x_top - width/2,
        top10['Social cost of carbon emissions (lower bound, in USD)'],
        width=width, label='Lower bound ($66/tCO2eq)', color='red', edgecolor='black'
    )
    bars_upper_top = ax_top.bar(
        x_top + width/2,
        top10['Social cost of carbon emissions (upper bound, in USD)'],
        width=width, label='Upper bound ($200/tCO2eq)', color='blue', edgecolor='black'
    )
    ax_top.bar_label(
        bars_lower_top,
        labels=[fmt_money_rounded(v) for v in top10['Social cost of carbon emissions (lower bound, in USD)']],
        fontsize=13, padding=2, rotation=0
    )
    ax_top.bar_label(
        bars_upper_top,
        labels=[fmt_money_rounded(v) for v in top10['Social cost of carbon emissions (upper bound, in USD)']],
        fontsize=13, padding=2, rotation=0
    )
    ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_money_rounded(v)))
    ax_top.set_xticks(x_top, top10['Display Name'], rotation=45, ha='right')
    ax_top.tick_params(axis='x', labelsize=13)
    ax_top.tick_params(axis='y', labelsize=13)
    ax_top.set_ylabel('Social cost of carbon emissions (USD)', fontsize=13)
    ax_top.set_title('Social cost of carbon emissions — Top 10 models (USD)', fontsize=15)
    ax_top.margins(x=0.02, y=0.25)
    ax_top.legend(fontsize=12)

    x_rest = np.arange(len(rest))
    bars_lower_rest = ax_rest.bar(
        x_rest - width/2,
        rest['Social cost of carbon emissions (lower bound, in USD)'],
        width=width, label='Lower bound ($66/tCO2eq)', color='red', edgecolor='black'
    )
    bars_upper_rest = ax_rest.bar(
        x_rest + width/2,
        rest['Social cost of carbon emissions (upper bound, in USD)'],
        width=width, label='Upper bound ($200/tCO2eq)', color='blue', edgecolor='black'
    )
    ax_rest.bar_label(
        bars_lower_rest,
        labels=[fmt_money_rounded(v) for v in rest['Social cost of carbon emissions (lower bound, in USD)']],
        fontsize=13, padding=2, rotation=90
    )
    ax_rest.bar_label(
        bars_upper_rest,
        labels=[fmt_money_rounded(v) for v in rest['Social cost of carbon emissions (upper bound, in USD)']],
        fontsize=13, padding=2, rotation=90
    )
    ax_rest.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_money_rounded(v)))
    ax_rest.set_xticks(x_rest, rest['Display Name'], rotation=45, ha='right')
    ax_rest.tick_params(axis='x', labelsize=13)
    ax_rest.tick_params(axis='y', labelsize=13)
    ax_rest.set_xlabel('Model names', fontsize=13)
    ax_rest.set_ylabel('Social cost of carbon emissions (USD)', fontsize=13)
    ax_rest.set_title('Social cost of carbon emissions — Remaining models (USD)', fontsize=15)
    ax_rest.margins(x=0.02, y=0.3)
    ax_rest.legend(fontsize=12)

    plt.tight_layout(pad=5.0, h_pad=16.0, rect=[0, 0, 1, 0.96])
    plt.show()

    x = np.arange(len(carbon_sorted))
    #Graph: Opportunity costs of training CO2 emissions — equivalent one-way LHR to JFK flights
    fig2, ax3 = plt.subplots(1, 1, figsize=(18, 10))
    fig2.subplots_adjust(bottom=0.45)
    bars4 = ax3.bar(x, opp_cost_co2_emissions_flights, width=0.5, color='blue', edgecolor='black')
    ax3.bar_label(bars4, labels=[f"{math.floor(v):,}" for v in opp_cost_co2_emissions_flights], fontsize=10, padding=2)
    ax3.set_xlabel('Model names')
    ax3.set_xticks(x, carbon_sorted['Display Name'], rotation=45, ha='right')
    ax3.tick_params(axis='x', labelsize=10)
    ax3.set_ylabel('Number of one-way LHR to JFK flights [British Airways]')
    ax3.set_title('Opportunity costs of models during training\n(one-way LHR to JFK flights [British Airways])')

    plt.tight_layout(pad=5.0)
    plt.show()

    #Graph: Opportunity costs of training CO2 emissions — equivalent Czech resident annual emissions
    fig3, ax4 = plt.subplots(1, 1, figsize=(18, 10))
    fig3.subplots_adjust(bottom=0.45)
    bars5 = ax4.bar(x, opp_cost_co2_emissions_czech_residents, width=0.35, color='green', edgecolor='black')
    ax4.bar_label(bars5, labels=[f"{math.floor(v):,}" for v in opp_cost_co2_emissions_czech_residents], fontsize=10, padding=2)
    ax4.set_xlabel('Model names')
    ax4.set_ylabel('Number of Czech residents')
    ax4.set_title('Opportunity costs of training emissions\n(equivalent Czech resident annual emissions)')
    ax4.set_xticks(x, carbon_sorted['Display Name'], rotation=45, ha='right')
    ax4.tick_params(axis='x', labelsize=10)

    plt.tight_layout(pad=5.0)
    plt.show()

def plot_training_data_water(water_df: pd.DataFrame, crop_prices_df: pd.DataFrame):
    #multiply the footprints from Mekonnen & Hoekstra (2011) by 1000 to convert to L/ton instead of m^3/ton
    corn_blue_water_footprint = 81 * 1000
    olive_oil_blue_water_footprint = 2388 * 1000
    bananas_blue_water_footprint = 97 * 1000
    wheat_blue_water_footprint = 342 * 1000

    #sort descending by water consumption, split top 10 vs remaining
    water_sorted = water_df.sort_values(
        'Estimated total water consumption (L)', ascending=False
    ).reset_index(drop=True)
    top10_w = water_sorted.iloc[:10].reset_index(drop=True)
    rest_w = water_sorted.iloc[10:].reset_index(drop=True)

    #Graph: Total water consumption during training — Top 10 models (upper subplot) and remaining models (lower subplot)
    fig_w, (ax_top_w, ax_rest_w) = plt.subplots(2, 1, figsize=(20, 22))

    bars_top = ax_top_w.bar(
        top10_w['Display Name'], top10_w['Estimated total water consumption (L)'],
        width=0.5, color='blue', edgecolor='black'
    )
    ax_top_w.bar_label(
        bars_top,
        labels=[fmt_count_rounded(v) for v in top10_w['Estimated total water consumption (L)']],
        fontsize=13, padding=2, rotation=0
    )
    ax_top_w.set_ylabel('Total water consumption (L)', fontsize=13)
    ax_top_w.set_title('Total water consumption during training — Top 10 models (L)', fontsize=15)
    ax_top_w.tick_params(axis='x', rotation=45, labelsize=13)
    ax_top_w.tick_params(axis='y', labelsize=13)
    plt.setp(ax_top_w.get_xticklabels(), ha='right')
    ax_top_w.margins(x=0.02, y=0.25)
    ax_top_w.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_count_rounded(v)))

    bars_rest = ax_rest_w.bar(
        rest_w['Display Name'], rest_w['Estimated total water consumption (L)'],
        width=0.5, color='blue', edgecolor='black'
    )
    ax_rest_w.bar_label(
        bars_rest,
        labels=[fmt_count_rounded(v) for v in rest_w['Estimated total water consumption (L)']],
        fontsize=13, padding=2, rotation=0
    )
    ax_rest_w.set_xlabel('LLM model name', fontsize=13)
    ax_rest_w.set_ylabel('Total water consumption (L)', fontsize=13)
    ax_rest_w.set_title('Total water consumption during training — Remaining models (L)', fontsize=15)
    ax_rest_w.tick_params(axis='x', rotation=45, labelsize=13)
    ax_rest_w.tick_params(axis='y', labelsize=13)
    plt.setp(ax_rest_w.get_xticklabels(), ha='right')
    ax_rest_w.margins(x=0.02, y=0.3)
    ax_rest_w.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_count_rounded(v)))

    plt.tight_layout(pad=4.0, h_pad=16.0, rect=[0, 0, 1, 0.96])
    plt.show()

    water_opportunity_costs_labels = ['Corn', 'Olive oil', 'Bananas', 'Wheat']
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

    #Graph: Water opportunity costs by crop — metric tons (left subplot) and monetized USD value (right subplot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    bars2 = ax1.bar(water_opportunity_costs_labels, water_opportunity_costs, width=0.5, color='blue')
    ax1.bar_label(bars2, labels=[fmt_count_rounded(v) for v in water_opportunity_costs], fontsize=10, padding=2)
    ax1.set_title('Metric tons of crops that could have been grown with\ntotal estimated water consumed during training')
    ax1.set_xlabel('Crops')
    ax1.set_ylabel('Amount of crops (in metric tons)')
    ax1.tick_params(axis='x', labelsize=10)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_count_rounded(v)))

    bars3 = ax2.bar(water_opportunity_costs_labels, water_opportunity_costs_monetized, width=0.5, color='blue')
    ax2.bar_label(bars3, labels=[fmt_money_rounded(v) for v in water_opportunity_costs_monetized], fontsize=10, padding=2)
    ax2.set_title('Monetized global average market cost of metric tons of crops\nthat could have been grown with total estimated water consumed during training')
    ax2.set_xlabel('Crops')
    ax2.set_ylabel('Monetized global average cost of crops on market (in USD)')
    ax2.tick_params(axis='x', labelsize=10)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_money_rounded(v)))

    plt.tight_layout(pad=2.0)
    plt.show()


def clean_training_data():
    os.makedirs("graphs", exist_ok=True)
    #import training data, remove unused columns, then call plotting functions
    training_df_carbon = pd.read_csv("carbon_training_data.csv")
    training_df_water = pd.read_csv("water_training_data.csv")
    crop_prices_df = pd.read_csv("global_price_of_crops.csv")
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

    plot_training_data_co2(training_df_carbon)
    plot_training_data_water(training_df_water, crop_prices_df)
