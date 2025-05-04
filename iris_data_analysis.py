# iris_data_analysis.py
# Complete data analysis of the Iris dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def load_and_explore_data():
    print("=== Task 1: Load and Explore the Dataset ===")
    try:
        iris = load_iris()
        df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                         columns=iris['feature_names'] + ['target'])
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        print("Dataset loaded successfully!")
        print(f"Shape of dataset: {df.shape}")
        
        # Display first few rows
        print("\nFirst 5 rows of data:")
        print(df.head())
        
        # Explore dataset structure
        print("\nData types:")
        print(df.dtypes)
        
        print("\nMissing values:")
        print(df.isnull().sum())
        
        return df
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def basic_data_analysis(df):
    print("\n=== Task 2: Basic Data Analysis ===")
    
    # Basic statistics
    print("\nBasic statistics for numerical columns:")
    print(df.describe())
    
    # Group by species and compute means
    print("\nMean measurements by species:")
    print(df.groupby('species').mean())
    
    # Interesting findings
    print("\nKey Observations:")
    print("1. Setosa has significantly smaller petal dimensions than other species")
    print("2. Virginica has the largest measurements on average")
    print("3. All species have similar sepal widths (around 3 cm)")

def create_visualizations(df):
    print("\n=== Task 3: Data Visualization ===")
    
    # Set style and figure size
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 1. Line chart (showing trends by index since we don't have time data)
    plt.subplot(2, 2, 1)
    df['sepal length (cm)'].plot(title='Sepal Length Trend', color='green')
    plt.ylabel('cm')
    plt.xlabel('Sample Index')
    
    # 2. Bar chart (average petal length by species)
    plt.subplot(2, 2, 2)
    sns.barplot(x='species', y='petal length (cm)', data=df)
    plt.title('Average Petal Length by Species')
    plt.ylabel('cm')
    
    # 3. Histogram (sepal width distribution)
    plt.subplot(2, 2, 3)
    sns.histplot(df['sepal width (cm)'], bins=15, kde=True, color='purple')
    plt.title('Sepal Width Distribution')
    plt.xlabel('cm')
    
    # 4. Scatter plot (sepal length vs petal length)
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', 
                    hue='species', data=df, palette='deep')
    plt.title('Sepal vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    
    plt.tight_layout()
    plt.show()
    
    # Additional visualizations
    print("\nCreating additional visualizations...")
    
    # Pairplot to show all relationships
    sns.pairplot(df, hue='species')
    plt.suptitle('Pairwise Relationships in Iris Dataset', y=1.02)
    plt.show()
    
    # Boxplot to show distribution by species
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='species', y='petal width (cm)', data=df)
    plt.title('Petal Width Distribution by Species')
    plt.ylabel('cm')
    plt.show()

def main():
    # Load and explore data
    df = load_and_explore_data()
    
    if df is not None:
        # Perform basic analysis
        basic_data_analysis(df)
        
        # Create visualizations
        create_visualizations(df)
        
        # Final observations
        print("\n=== Final Findings ===")
        print("1. The three iris species are clearly separable based on petal measurements")
        print("2. Setosa has the most distinct characteristics with smaller petals")
        print("3. Versicolor and Virginica show some overlap in sepal measurements")
        print("4. Petal measurements are more reliable for species classification than sepal measurements")

if __name__ == "__main__":
    main()