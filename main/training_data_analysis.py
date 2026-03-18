import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np

def fmt_val(v):
    if abs(v) < 10000:
        return f"{v:.1f}"
    return f"{v:.1e}"

#function that labels scatterplots and adjusts the text labels
def label_scatterplots(lower=None, upper=None, carbon_df=None, water_df=None, water_consumption=None):
    scatterplot_texts = []

    #check if the values for carbon emissions are set or if the values for water consumption are set.
    #if the values for carbon emissions are set, run this line of code to add the labels to each of the scatterplot points
    if lower is not None and upper is not None and carbon_df is not None:
        for i,model in enumerate(carbon_df['LLM model']):
            scatterplot_texts.append(plt.text(carbon_df['Parameters (billions)'].iloc[i],
             lower.iloc[i],
             model)
            )

            scatterplot_texts.append(plt.text(carbon_df['Parameters (billions)'].iloc[i],
             upper.iloc[i],
             model)
            )

        adjust_text(scatterplot_texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    else:
        for i,model in enumerate(water_df['LLM model']):
            scatterplot_texts.append(plt.text(water_df['Parameters (billions)'].iloc[i], water_consumption.iloc[i], model))

        adjust_text(scatterplot_texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

def plot_training_data_co2(carbon_df: pd.DataFrame):
    #plot estimated yearly total CO2 emissions during training
    carbon_df['Social cost of carbon emissions (lower bound, in USD)'] = carbon_df['CO2 (tCO2eq)']*66
    carbon_df['Social cost of carbon emissions (upper bound, in USD)'] = carbon_df['CO2 (tCO2eq)']*200

    print(f'Social costs of carbon emissions (lower bound): {','.join(str(i) for i in carbon_df['Social cost of carbon emissions (lower bound, in USD)'])}')
    print(f'Social costs of carbon emissions (upper bound): {','.join(str(i) for i in carbon_df['Social cost of carbon emissions (upper bound, in USD)'])}')

    #get the index of the minimum of the lower bound values and then print out the minimum social cost for the lower bound
    #also print out the model name associated with it
    min_idx_lower_bound = carbon_df["Social cost of carbon emissions (lower bound, in USD)"].idxmin()
    print(f'Min social cost of carbon emissions (lower bound)' 
        f'{carbon_df["Social cost of carbon emissions (lower bound, in USD)"].min()}', 
        f'name: {carbon_df.loc[min_idx_lower_bound,"LLM model"]}'
    )

    #same thing but for upper bound
    min_idx_upper_bound = carbon_df["Social cost of carbon emissions (upper bound, in USD)"].idxmin()
    print(f'Min social cost of carbon emissions (upper bound)' 
        f'{carbon_df["Social cost of carbon emissions (upper bound, in USD)"].min()}', 
        f'name: {carbon_df.loc[min_idx_upper_bound,"LLM model"]}'
    )

    years_released =  list(range(carbon_df['Year released'].min(),carbon_df['Year released'].max()+1))

    #plot social cost of carbon emissions during training vs. parameter count in a scatter plot
    plt.figure(figsize=(22,6))
    plt.scatter(carbon_df['Parameters (billions)'],carbon_df['Social cost of carbon emissions (lower bound, in USD)'], color='red',label='Lower bound emissions social cost ($66)')
    plt.scatter(carbon_df['Parameters (billions)'],carbon_df['Social cost of carbon emissions (upper bound, in USD)'], color='blue', label='Upper bound emissions social cost ($200)')
    plt.xlabel('Parameters (billions)')
    plt.ylabel('Social cost of carbon emissions (in USD)')
    plt.title('Social cost of carbon emissions in USD across all models vs. parameter count (in billions) during training')

    label_scatterplots(carbon_df['Social cost of carbon emissions (lower bound, in USD)'],
     carbon_df['Social cost of carbon emissions (upper bound, in USD)'],
     carbon_df,None,None)
    plt.legend()
    plt.show()

    sf_ny_roundtrip_flights_co2_tons_total = 180.4
    czech_residents_emissions_tons_per_capita = 7.04

    opp_cost_co2_emissions_flights = []
    opp_cost_co2_emissions_czech_residents = []

    for i in carbon_df['CO2 (tCO2eq)']:
        opp_cost_co2_emissions_flights.append(i/sf_ny_roundtrip_flights_co2_tons_total)
        opp_cost_co2_emissions_czech_residents.append(i/czech_residents_emissions_tons_per_capita)

    #plot social cost of carbon emissions of all models for lower bound and upper bound SCC values
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(42,11))
    x = np.arange(len(carbon_df['LLM model']))
    width = 0.45
    fig.subplots_adjust(bottom=0.45)

    ax1.bar(x - width/2, carbon_df['Social cost of carbon emissions (lower bound, in USD)'], width=width, label='Lower bound ($66)', color='red', edgecolor='black')
    ax1.bar(x + width/2, carbon_df['Social cost of carbon emissions (upper bound, in USD)'], width=width, label='Upper bound ($200)', color='blue',edgecolor='black')

    ax1.set_xticks(x, carbon_df['Display Name'],rotation=45,ha='right')
    ax1.tick_params(axis='x', labelsize=7)
    ax1.set_xlabel('Model names')
    ax1.set_ylabel('Social cost of carbon emissions in USD')
    ax1.set_title('Social cost of carbon emissions of all models in dataset in USD')
    ax1.legend()

    bars3 = ax2.bar(x,carbon_df['CO2 (tCO2eq)'],width=0.9,color='blue',edgecolor='black')
    ax2.bar_label(bars3, fmt='%.3g', fontsize=6, padding=2)
    ax2.set_xlabel('Model names')
    ax2.set_xticks(x,carbon_df['Display Name'],rotation=45,ha='right')
    ax2.tick_params(axis='x', labelsize=7)
    ax2.set_ylabel('CO2 emissions (tCO2eq) during training')
    ax2.set_title('CO2 emissions (tCO2eq) during training per model')

    fig2, (ax3,ax4) = plt.subplots(1,2,figsize=(36,8))
    fig2.subplots_adjust(bottom=0.45)
    bars4 = ax3.bar(x,opp_cost_co2_emissions_flights,width=0.9,color='blue',edgecolor='black')
    ax3.bar_label(bars4, fmt='%.3g', fontsize=6, padding=2)
    ax3.set_xlabel('Model names')
    ax3.set_xticks(x,carbon_df['Display Name'],rotation=45,ha='right')
    ax3.tick_params(axis='x', labelsize=7)
    ax3.set_ylabel('Number of roundtrip SFO to JFK flights')
    ax3.set_title('Opportunity costs of models during training \n (roundtrip SFO to JFK flights)')

    bars5 = ax4.bar(x,opp_cost_co2_emissions_czech_residents,width=0.9,color='green',edgecolor='black')
    ax4.bar_label(bars5, fmt='%.3g', fontsize=6, padding=2)
    ax4.set_xlabel('Model names')
    ax4.set_ylabel('Number of czech households')
    ax4.set_title('Opportunity costs of training scenarios \n (number of czech households)')
    ax4.set_xticks(x,carbon_df['Display Name'],rotation=45,ha='right')
    ax4.tick_params(axis='x', labelsize=7)

    plt.tight_layout(pad=5.0)
    plt.show()

def plot_training_data_water(water_df: pd.DataFrame, crop_prices_df: pd.DataFrame):
    #multiply the footprints from Mekonnen & Hoekstra (2011) by 1000 to convert to L/ton instead of m^3/ton
    corn_blue_water_footprint = 81 * 1000
    olive_oil_blue_water_footprint = 2388 * 1000
    bananas_blue_water_footprint = 97 * 1000
    wheat_blue_water_footprint = 342 * 1000

    #plot total parameters vs. estimated water consumption in liters (scatterplot)
    plt.figure(figsize=(15,8))
    plt.scatter(water_df['Parameters (billions)'],water_df['Estimated total water consumption (L)'],color='blue')
    plt.xlabel('Parameters (billions)')
    plt.ylabel('Total water consumption (L)')
    plt.title('Total water consumption of models during training (L) vs. parameter count (in billions)')
    label_scatterplots(water_df=water_df,water_consumption=water_df['Estimated total water consumption (L)'])
    plt.show()

    #plot total water consumption of all models in the training dataset (bar chart)
    plt.figure(figsize=(15,8))
    bars1 = plt.bar(water_df['Display Name'],water_df['Estimated total water consumption (L)'],color='blue')
    plt.gca().bar_label(bars1, fmt='%.3g', fontsize=6, padding=2)
    plt.xlabel('LLM model name')
    plt.ylabel('Total water consumption (L)')
    plt.title('Total water consumption of models during training (in L) for each model in dataset')
    plt.xticks(rotation=45,ha='right')
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

    print(f'Water opportunity costs: {", ".join(str(i) for i in water_opportunity_costs)}')
    print(f'Water opportunity costs monetized value: ${", ".join(str(i) for i in water_opportunity_costs_monetized)}')

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


def plot_training_data_boxplots(carbon_df: pd.DataFrame, water_df: pd.DataFrame):
    #box and whisker plots for CO2 emissions, water consumption, and social costs of carbon during training
    #outlier points are labelled with the model name and exact value
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 9))

    def label_outliers(ax, bp, box_index, series, display_names):
        #annotate each outlier point with its model name and value, using adjust_text to avoid collisions
        texts = []
        for xval, yval in zip(bp['fliers'][box_index].get_xdata(), bp['fliers'][box_index].get_ydata()):
            match = display_names[np.isclose(series, yval)]
            for name in match:
                texts.append(ax.text(xval, yval, f"{name}\n({fmt_val(yval)})", fontsize=7))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    def label_stats(ax, bp, box_index, series, display_names):
        #annotate the median, min/max whisker caps, and Q3 with their values and closest model names
        x_right = bp['medians'][box_index].get_xdata()[1]
        median_val = bp['medians'][box_index].get_ydata()[0]
        min_val = bp['caps'][2 * box_index].get_ydata()[0]
        max_val = bp['caps'][2 * box_index + 1].get_ydata()[0]
        q1_val = series.quantile(0.25)
        q3_val = series.quantile(0.75)

        def lookup(val):
            matches = display_names[np.isclose(series, val)]
            return f'\n({", ".join(matches)})' if len(matches) > 0 else ''

        #avoid duplicate label if the absolute max is an outlier (already labeled by label_outliers)
        if series.max() > max_val:
            max_name = lookup(max_val)
        else:
            max_name = f'\n({display_names[series.idxmax()]})'

        iqr = q3_val - q1_val
        ax.text(x_right, median_val,         f' Median: {fmt_val(median_val)}{lookup(median_val)}', fontsize=7, va='center')
        ax.text(x_right, min_val - iqr * 0.15, f' Min: {fmt_val(min_val)}{lookup(min_val)}',        fontsize=7, va='center')
        ax.text(x_right, max_val,    f' Max: {fmt_val(max_val)}{max_name}',                  fontsize=7, va='center')
        q1_model_name = display_names[(series - q1_val).abs().idxmin()]
        q3_model_name = display_names[(series - q3_val).abs().idxmin()]
        ax.text(x_right, q1_val,     f' Q1: {fmt_val(q1_val)}\n({q1_model_name})',            fontsize=7, va='center')
        ax.text(x_right, q3_val,     f' Q3: {fmt_val(q3_val)}\n({q3_model_name})',            fontsize=7, va='center')

    #CO2 emissions box plot
    bp1 = ax1.boxplot(carbon_df['CO2 (tCO2eq)'], patch_artist=True, boxprops=dict(facecolor='steelblue', alpha=0.5))
    label_outliers(ax1, bp1, 0, carbon_df['CO2 (tCO2eq)'], carbon_df['Display Name'])
    label_stats(ax1, bp1, 0, carbon_df['CO2 (tCO2eq)'], carbon_df['Display Name'])
    ax1.set_ylabel('CO2 emissions (tCO2eq)', fontsize=11)
    ax1.set_xlabel('All models in dataset', fontsize=10)
    ax1.set_title('Distribution of total CO2 emissions\nduring training (tCO2eq)', fontsize=12)
    ax1.set_xticks([])

    #water consumption box plot
    bp2 = ax2.boxplot(water_df['Estimated total water consumption (L)'], patch_artist=True, boxprops=dict(facecolor='mediumseagreen', alpha=0.5))
    label_outliers(ax2, bp2, 0, water_df['Estimated total water consumption (L)'], water_df['Display Name'])
    label_stats(ax2, bp2, 0, water_df['Estimated total water consumption (L)'], water_df['Display Name'])
    ax2.set_ylabel('Total water consumption (L)', fontsize=11)
    ax2.set_xlabel('All models in dataset', fontsize=10)
    ax2.set_title('Distribution of total water\nconsumption during training (L)', fontsize=12)
    ax2.set_xticks([])

    #social cost of carbon box plot with two boxes side by side (lower and upper SCC bound)
    #each box is colored individually since patch_artist only supports a single boxprops color
    bp3 = ax3.boxplot(
        [carbon_df['Social cost of carbon emissions (lower bound, in USD)'],
         carbon_df['Social cost of carbon emissions (upper bound, in USD)']],
        labels=['Lower bound\n($66/tCO2eq)', 'Upper bound\n($200/tCO2eq)'],
        patch_artist=True,
    )
    bp3['boxes'][0].set_facecolor('salmon')
    bp3['boxes'][0].set_alpha(0.5)
    bp3['boxes'][1].set_facecolor('mediumpurple')
    bp3['boxes'][1].set_alpha(0.5)
    label_outliers(ax3, bp3, 0, carbon_df['Social cost of carbon emissions (lower bound, in USD)'], carbon_df['Display Name'])
    label_outliers(ax3, bp3, 1, carbon_df['Social cost of carbon emissions (upper bound, in USD)'], carbon_df['Display Name'])
    label_stats(ax3, bp3, 0, carbon_df['Social cost of carbon emissions (lower bound, in USD)'], carbon_df['Display Name'])
    label_stats(ax3, bp3, 1, carbon_df['Social cost of carbon emissions (upper bound, in USD)'], carbon_df['Display Name'])
    ax3.set_ylabel('Social cost of carbon emissions (USD)', fontsize=11)
    ax3.set_xlabel('Social cost of carbon bounds', fontsize=10)
    ax3.set_title('Distribution of social cost of carbon\nemissions during training (USD)', fontsize=12)
    ax3.tick_params(axis='x', labelsize=9)

    plt.tight_layout(pad=3.0)
    plt.show()


def clean_training_data():
    #import training data, remove unused columns, then call plotting functions
    training_df_carbon = pd.read_csv("carbon_training_data.csv")
    training_df_water = pd.read_csv("water_training_data.csv")
    crop_prices_df = pd.read_csv("global_price_of_crops.csv")
    
    training_df_carbon = training_df_carbon.drop(columns=['Source',
     'Country of the organization(s) that trained the model',
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
    plot_training_data_water(training_df_water,crop_prices_df)
    plot_training_data_boxplots(training_df_carbon,training_df_water)