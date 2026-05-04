# Bachelor's Thesis Data Analysis

This repository contains the data analysis source code for my Bachelor's Thesis at Charles University. The project quantifies the environmental costs (CO2 emissions and water consumption) of large language models (LLMs) during both training and inference, and expresses those costs in terms of social cost of carbon and real-world opportunity costs (e.g. equivalent transatlantic flights, crop production).

## What it does

The analysis is split into two parts:

**Training analysis** (`training_data_analysis.py`)
- Plots CO2 emissions and their social cost (lower bound: $66/tCO2eq, upper bound: $200/tCO2eq) for a dataset of LLMs during training
- Plots water consumption during training and its opportunity cost in crop equivalents (corn, olive oil, bananas, wheat)

**Inference analysis** (`inference_data_analysis.py`)
- Estimates monthly CO2 and water costs for open-source models (sourced from the Hugging Face OpenLLM leaderboard and download counts) across multiple daily-active-user and prompts-per-day scenarios
- Plots the same metrics for proprietary models (ChatGPT and Google AI Overviews/Gemini) using publicly available figures

## Installation

**Requirements:** Python 3.10+

1. Clone the repository:
   ```bash
   git clone https://github.com/benw49/bachelors-thesis-data-analysis-python.git
   cd bachelors-thesis-data-analysis-python
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Ensure the following data files are present in the `main/` directory:
- `carbon_training_data.csv` — training CO2 emissions data
- `water_training_data.csv` — training water consumption data
- `energy_mixes.csv`
- `global_price_of_crops.csv`
- `top-models-by-downloads.csv`
- `openllm_leaderboard.csv`

Then run:
```bash
cd main
python main.py
```

This will display all graphs one by one in the matplotlib interface. To save a graph, use the save icon in the matplotlib toolbar before closing each window.
