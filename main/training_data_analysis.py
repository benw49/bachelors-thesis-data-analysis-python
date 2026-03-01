import pandas as pd
import matplotlib.pyplot as plt 
from adjustText import adjust_text
import numpy as np

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
            scatterplot_texts.append(plt.text(water_df['Parameters (billions)'].iloc[i],water_consumption.iloc[i],model))
        
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

    #plot social cost of carbon emissions of all models for lower bound and upper bound SCC values
    x = np.arange(len(carbon_df['LLM model']))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, carbon_df['Social cost of carbon emissions (lower bound, in USD)'], width=width, label='Lower bound ($66)', color='red', edgecolor='black')
    plt.bar(x + width/2, carbon_df['Social cost of carbon emissions (upper bound, in USD)'], width=width, label='Upper bound ($200)', color='blue',edgecolor='black')
    
    plt.xticks(x, carbon_df['LLM model'],rotation=45,ha='right')
    plt.xlabel('Model names')
    plt.ylabel('Social cost of carbon emissions in USD')
    plt.title('Social cost of carbon emissions of all models in dataset in USD')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_training_data_water(water_df: pd.DataFrame, crop_prices_df: pd.DataFrame):
    #multiply the footprints from Mekonnen & Hoekstra (2011) by 1000 to convert to L/ton instead of m^3/ton
    corn_blue_water_footprint = 81 * 1000
    olive_oil_blue_water_footprint = 2388 * 1000
    bananas_blue_water_footprint = 97 * 1000
    wheat_blue_water_footprint = 342 * 1000

    #plot total parameters vs. estimated water consumption in liters
    plt.figure(figsize=(15,8))
    plt.scatter(water_df['Parameters (billions)'],water_df['Estimated total water consumption (L)'],color='blue')
    plt.xlabel('Parameters (billions)')
    plt.ylabel('Total water consumption (L)')
    plt.title('Total water consumption of models during training (L) vs. parameter count (in billions)')
    label_scatterplots(water_df=water_df,water_consumption=water_df['Estimated total water consumption (L)'])
    plt.show()

    #plot total water consumption of all models in the training dataset
    plt.figure(figsize=(15,8))
    plt.bar(water_df['LLM model'],water_df['Estimated total water consumption (L)'],color='blue')
    plt.xlabel('LLM model name')
    plt.ylabel('Total water consumption (L)')
    plt.title('Total water consumption of models during training (in L) for each model in dataset')
    plt.xticks(rotation=45,ha='right')
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
        (total_water_consumed_training / olive_oil_blue_water_footprint)*crop_prices_df['Yearly Average in USD per metric ton (2025)'].iloc[1],
        (total_water_consumed_training / bananas_blue_water_footprint)*crop_prices_df['Yearly Average in USD per metric ton (2025)'].iloc[2],
        (total_water_consumed_training / wheat_blue_water_footprint)*crop_prices_df['Yearly Average in USD per metric ton (2025)'].iloc[3]
    ]

    print(f'Water opportunity costs: {", ".join(str(i) for i in water_opportunity_costs)}')
    print(f'Water opportunity costs monetized value: ${", ".join(str(i) for i in water_opportunity_costs_monetized)}')

    #plot water opportunity costs
    plt.figure(figsize=(15,8))
    plt.bar(water_opportunity_costs_labels,water_opportunity_costs,width=0.5,color='blue')
    plt.title('Tons of crops that could have been grown with total estimated water consumed during training')
    plt.xlabel('Crops')
    plt.ylabel('Amount of crops (in tons)')
    plt.show()

    #plot monetized value of opportunity costs 
    plt.figure(figsize=(15,8))
    plt.bar(water_opportunity_costs_labels,water_opportunity_costs_monetized,width=0.5,color='blue')
    plt.title('Monetized global average market cost of tons of crops that could have been grown with total estimated water consumed during training')
    plt.xlabel('Crops')
    plt.ylabel('Monetized global average cost of crops on market (in USD)')
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

    plot_training_data_co2(training_df_carbon)
    plot_training_data_water(training_df_water,crop_prices_df)