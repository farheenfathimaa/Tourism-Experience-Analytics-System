import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("--- STEP 1: Loading Cleaned Data ---")
    if not os.path.exists("cleaned_data.csv"):
        raise FileNotFoundError("cleaned_data.csv not found. Please run data_preprocessing.py first.")
        
    df = pd.read_csv("cleaned_data.csv")
    print(f"Data loaded successfully. Shape: {df.shape}")

    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Plots will be saved to: {os.path.abspath(plots_dir)}")

    # Set visualization style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16})

    # Color Palette: Modern and cohesive HSL-like colors
    palette = sns.color_palette("viridis", 8)

    print("\n--- STEP 2: Generating and Saving Plots ---")

    # Plot 1: Rating distribution (histogram)
    plt.figure(figsize=(8, 5))
    # Count of each rating
    sns.countplot(data=df, x='Rating', palette="Blues_d")
    plt.title("Rating Distribution")
    plt.xlabel("Rating (1-5)")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rating_distribution.png"), dpi=150)
    plt.close()
    print("Saved rating_distribution.png")

    # Plot 2: Visit mode distribution (bar chart)
    plt.figure(figsize=(8, 5))
    visit_mode_counts = df['VisitMode'].value_counts()
    sns.barplot(x=visit_mode_counts.index, y=visit_mode_counts.values, palette="rocket")
    plt.title("Visit Mode Distribution")
    plt.xlabel("Visit Mode")
    plt.ylabel("Number of Visits")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "visit_mode_distribution.png"), dpi=150)
    plt.close()
    print("Saved visit_mode_distribution.png")

    # Plot 3: Top 10 most visited attractions (bar chart)
    plt.figure(figsize=(10, 6))
    top_attractions = df['Attraction'].value_counts().head(10)
    sns.barplot(x=top_attractions.values, y=top_attractions.index, palette="mako")
    plt.title("Top 10 Most Visited Attractions")
    plt.xlabel("Number of Visits")
    plt.ylabel("Attraction Name")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "top_attractions.png"), dpi=150)
    plt.close()
    print("Saved top_attractions.png")

    # Plot 4: User distribution by Continent (pie or bar)
    plt.figure(figsize=(8, 8))
    continent_counts = df['Continent'].value_counts()
    # If there is "Unknown", let's display it too or filter it, but we kept it.
    plt.pie(
        continent_counts.values, 
        labels=continent_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=sns.color_palette("pastel", len(continent_counts))
    )
    plt.title("User Distribution by Continent")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "user_continent_distribution.png"), dpi=150)
    plt.close()
    print("Saved user_continent_distribution.png")

    # Plot 5: Monthly visit trends (line chart)
    plt.figure(figsize=(9, 5))
    monthly_counts = df['VisitMonth'].value_counts().sort_index()
    # Map month numbers to names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    sns.lineplot(x=monthly_counts.index, y=monthly_counts.values, marker="o", color="teal", linewidth=2.5)
    plt.title("Monthly Visit Trends")
    plt.xlabel("Month")
    plt.ylabel("Number of Visits")
    plt.xticks(ticks=range(1, 13), labels=month_names)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "monthly_trends.png"), dpi=150)
    plt.close()
    print("Saved monthly_trends.png")

    # Plot 6: Average rating by attraction type (bar chart)
    plt.figure(figsize=(10, 6))
    avg_rating_type = df.groupby('AttractionType')['Rating'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_rating_type.values, y=avg_rating_type.index, palette="flare")
    plt.title("Average Rating by Attraction Type")
    plt.xlabel("Average Rating (1-5)")
    plt.ylabel("Attraction Type")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "avg_rating_by_type.png"), dpi=150)
    plt.close()
    print("Saved avg_rating_by_type.png")

    # Plot 7: Correlation heatmap of numerical features
    plt.figure(figsize=(8, 6))
    num_cols = ['Rating', 'VisitYear', 'VisitMonth', 'UserAvgRating', 'UserVisitCount']
    corr_matrix = df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print("Saved correlation_heatmap.png")

    # Plot 8: Rating distribution by visit mode (boxplot)
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x='VisitMode', y='Rating', palette="Set2")
    plt.title("Rating Distribution by Visit Mode")
    plt.xlabel("Visit Mode")
    plt.ylabel("Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rating_by_visit_mode_boxplot.png"), dpi=150)
    plt.close()
    print("Saved rating_by_visit_mode_boxplot.png")

    print("\n--- STEP 3: Key Statistics ---")
    
    # Compute stats
    popular_attraction = df['Attraction'].value_counts().index[0]
    popular_attraction_count = df['Attraction'].value_counts().values[0]
    
    common_visit_mode = df['VisitMode'].value_counts().index[0]
    common_visit_mode_count = df['VisitMode'].value_counts().values[0]
    
    avg_rating_overall = df['Rating'].mean()

    print("\n" + "="*50)
    print("KEY STATISTICS")
    print("="*50)
    print(f"Most Popular Attraction: {popular_attraction} ({popular_attraction_count} visits)")
    print(f"Most Common Visit Mode:  {common_visit_mode} ({common_visit_mode_count} visits)")
    print(f"Average Rating Overall:  {avg_rating_overall:.3f}")
    print("="*50)

if __name__ == "__main__":
    main()
