# Mushroom Classification Project

## Overview
This project analyzes the UCI Mushroom Dataset to develop classification models that can distinguish between edible and poisonous mushrooms. The goal is to identify which characteristics are most predictive of mushroom edibility and to compare the performance of different machine learning algorithms.

## Dataset
The dataset contains 8,124 mushroom samples with 23 categorical features, including cap shape, odor, gill size, and habitat. Each mushroom is classified as either edible or poisonous. The data is sourced from the UCI Machine Learning Repository available below:
https://archive.ics.uci.edu/dataset/73/mushroom


## Project Structure
The project is organized as follows:
- `data/` - Contains the dataset files
- `classification_mushrooms.ipynb` - Main Jupyter notebook with all analysis
- `README.md` - Project documentation

## Approach
The analysis follows these key steps:
1. **Data Exploration and Preparation**
   - Visual analysis of feature distributions by edibility
   - Handling missing values
   - Encoding categorical features

2. **Feature Selection**
   - Case 1: Odor only
   - Case 2: Spore print color only
   - Case 3: Spore print color + gill color
   - Case 4: Bruises + habitat

3. **Model Training and Evaluation**
   - Decision Trees and Logistic Regression for Case 1
   - Decision Trees and Random Forests for Case 2
   - Random Forests and Logistic Regression for Case 3
   - Random Forests and Decision Trees for Case 4
   - Comparison of model performance using accuracy, precision, recall, and F1-score

## Key Findings
- Certain mushroom characteristics are extremely predictive of edibility, particularly Odor
- Odor alone can classify mushrooms with 98.32% accuracy using a Decision Tree
- Models consistently avoided false negatives (never classified poisonous mushrooms as edible)
- Decision Trees performed exceptionally well, suggesting the classification follows clear rules
- Simple models with few features achieved comparable performance to more complex approaches

## Setup Instructions
To run this project locally:

1. Clone this repository
git clone https://github.com/yourusername/mushroom-classification.git
cd mushroom-classification


2. Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. Within Virtual Environment install dependencies
pip install -r requirements.txt


4. Open `classification_mushrooms.ipynb` in your VSCode

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook extension in VSCode

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- UCI Machine Learning Repository for providing the mushroom dataset
- This project was completed as part of a machine learning classification assignment. Thanks to Dr. Denise Case.
- https://github.com/denisecase/ml-04/blob/main/CLASSIFICATION_PROJECT.md