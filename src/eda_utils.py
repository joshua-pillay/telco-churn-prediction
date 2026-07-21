"""
Module containing helper functions used for performing exploratory data analysis 
on the raw Telco Customer Churn dataset.
"""
#---------------------------------------------------------------------------------------
# Imports necessary libraries: `pandas` for data computation and manipulation;
# `pyplot` `ticker`, `Patch`, `LinearSegmentedColormap` from `matplotlib` and 
# `seaborn` for data visualisation:

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt                        # chart rendering
import matplotlib.ticker as mticker                    # chart axis formatting
from matplotlib.patches import Patch                   # global legend formatting
#---------------------------------------------------------------------------------------
# Provides a comma separator option for global formatting of axis tick labels, 
# which ensures better readability:

label_fmt = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}") 
#---------------------------------------------------------------------------------------
# Defines a handle with labelled visual markers for global legend formatting in 
# specified charts:    

bar_handles = [Patch(facecolor = "#8B0000", label = "Yes"), 
               Patch(facecolor = "#4472C4", label = "No")] 
#---------------------------------------------------------------------------------------
def confirm_categories(original_df, features, new_df_name):
    """
    Extracts specified features from a dataframe into a separate named
    dataframe, and verifies the number of categories for each feature.

    Parameters
    ----------
    original_df : pandas.Dataframe 
        Original feature data.
    features : list of str
        Names of features.
    new_df_name : str      
        Name of new dataframe.
          
    Returns
    -------
    cat_df : pandas.Dataframe 
        Extracted feature data.            
    """
    # makes a copy of features list to avoid global modification:
    feature_names = features.copy()
    
    churn = "Churn"
    feature_names.append(churn)
    
    cat_df = original_df[feature_names]

    # confirms the number of categories for each feature in the created dataframes:
    cat_unique = cat_df.drop(columns = ["Churn"]).nunique()

    print(f"Number of categories for each feature in {new_df_name}:\n", 
          cat_unique, sep = "")

    return cat_df
#---------------------------------------------------------------------------------------
def compute_count_percentages(feature_df):
    """
    Determines total value counts and corresponding percentage proportions for 
    each label in all categorical features of a specified dataframe.

    Parameters
    ---------- 
    feature_df : pandas.DataFrame 
        Categorical features, their label entries and the corresponding target 
        class.
        
    Returns
    ------- 
    counts : dict 
        Value counts for every category-class pair of each feature, stored 
        in dataframes. 
    percentages : dict
        Percentage proportions for every churning category-class pair of each 
        feature, stored in dataframes.       
    """
    # extracts feature names:
    columns = feature_df.columns.tolist()
    
    # removes target column name 'Churn':
    del columns[-1]             

    # computes label counts for every category-class pair per feature, with 
    # unstack() for pivoting class index from row to column:
    counts = {feature: feature_df.groupby([feature, "Churn"])["Churn"]
              .count()
              .unstack() for feature in columns}

    # computes label proportions of every category-class pair for churners only:
    props = {feature: (feature_df.groupby(feature)["Churn"]
                       .apply(lambda churn: (churn == "Yes").mean()) * 100)
             .round(2) for feature in columns}

    # appends '%' symbol to proportions in two_cat_pcnt_prop:
    percentages = {feature: props[feature].map(lambda pcnt: f"{pcnt}%") 
                   for feature, series in props.items()}

    return counts, percentages
#---------------------------------------------------------------------------------------
def plot_cat_bars(counts, percentages, feature, axes, y_label):
    """
    Produces formatted horizontal bar chart displaying the distribution of 
    category labels for a singular feature with a specified axes object 
    and y-axis label.

    Parameters
    ---------- 
    counts : dict
        Value counts for every category-class pair of each feature, stored 
        in dataframes. 
    percentages : dict 
        Percentage proportions for every churning category-class pair of 
        each feature, stored in dataframes. 
    feature : str
        Name of feature.
    axes : ax
        Axes object for bar plot. 
    y_label : str 
        Label of y-axis.
    """
    counts[feature].plot(kind = "barh",
                         stacked = True,
                         ax = axes,
                         color = {"Yes": "#8B0000", "No": "#4472C4"},
                         width = 0.4,
                         legend = False)

    # annotates each category bar with percentage proportion of churners: 
    axes.bar_label(axes.containers[1], labels = percentages[feature].values, 
                   padding = 4, fontsize = 10)

    axes.xaxis.set_major_formatter(label_fmt)             
    axes.tick_params(axis = "both", labelsize = 10)
    axes.set_xlabel("Number of Customers", fontsize = 11)
    axes.set_ylabel(y_label, fontsize = 11)
    axes.set_title(feature, fontsize = 13)
#---------------------------------------------------------------------------------------
def plot_features_grid(counts, percentages, nrows, ncols, fig_size, y_label, 
                       global_title, filename):
    """
    Produces and saves (to file) a formatted visual comprising horizontal 
    bar charts that display the distribution of labels for several features 
    with specified axes objects and y-axis labels.

    Parameters
    ---------- 
    counts : dict 
        Value counts for every category-class pair of each feature, stored 
        in dataframes. 
    percentages : dict 
        Percentage proportions for every churning category-class pair of
        each feature, stored in dataframes. 
    nrows : int 
        Number of rows in subplot grid.
    ncols : int 
        Number of columns in subplot grid.
    fig_size : tuple of int
        (width, height) dimensions of plot.
    y_label :str/list of str 
        Label/s of subplot y-axes.
    global_title : str  
        Global title of figure.
    filename : str
        Name of visual saved to file.
    """
    figure, axes = plt.subplots(nrows, ncols, figsize = fig_size, 
                                layout = "constrained", 
                                sharex = True)

    # specifies that if 'y_label' is a singular string, each subplot shares  
    # the same y-axis label. Otherwise, each subplot y-axis label is assigned
    # a string from a list of strings:
    
    if isinstance(y_label, str):
       # iteratively produces bars and formatting for each feature via 
       # plot_cat_bars, with flatten() for transforming rows of axes into 
       # individual axes objects: 
       for feature, ax in zip(list(counts.keys()), axes.flatten()):
           plot_cat_bars(counts, percentages, feature, ax, y_label)
    else:
        for feature, ax, label in zip(list(counts.keys()), axes.flatten(), y_label):
            plot_cat_bars(counts, percentages, feature, ax, label) 
            
    # places global legend below subplots at a fixed distance of 0.5 inches 
    # below bottom axes:

    figure.legend(handles = bar_handles, loc = "outside lower center", 
                  bbox_to_anchor = (0.5, -0.5 / fig_size[1]), 
                  ncols = 2, title = "Churn", title_fontsize = 12)

    # adds global figure title:
    plt.suptitle(global_title, fontsize = 18, fontweight = "bold")
    plt.tight_layout()
    plt.savefig(f"reports/figures/{filename}.png", dpi = 300, bbox_inches = 'tight')
    plt.show()    
#---------------------------------------------------------------------------------------
def plot_hist(feature_df, feature, bin_edges, x_label, axes, rotation, 
              xlabel_bins = None):
    """
    Produces a formatted histogram for a binned numeric feature with 
    specified bin edges and axes.

    Parameters
    ---------- 
    feature_df : pandas.DataFrame
        Numeric features, their label entries and the corresponding 
        target class.
    feature : str
        Name of feature.
    bin_edge : array of int
        Numeric bin edges.
    x_label : str
        Label of x-axis.
    axes : ax
        Axes object for histogram plot. 
    rotation : int
        Rotation angle of x-ticks.
    xlabel_bins : array of int
        Numeric labels for x-ticks. Default is None. 
    """
    sns.histplot(data = feature_df,
                 x = feature,
                 hue = "Churn",
                 ax = axes, 
                 multiple = "stack",
                 palette = {"Yes": "#8B0000", "No": "#4472C4"},
                 bins = bin_edges,
                 alpha = 1,
                 legend = False)

    # specifies that if xlabel_bins = None, then the labels for 
    # x-ticks are equivalent to the bin edges:
    if xlabel_bins is None:
       xlabel_bins = bin_edges
    
    # plot formatting:
    axes.tick_params(axis = "both", labelsize = 11)
    axes.set_xticks(xlabel_bins)
    axes.tick_params(axis = "x", rotation = rotation)
    axes.set_xlabel(x_label, fontsize = 12)
    axes.set_ylabel("Number of Customers", fontsize = 12)
    axes.set_title(feature, fontsize = 13)    
#---------------------------------------------------------------------------------------
def assign_numeric_entry(feature_df, feature, edges):
    """
    Defines bin labels for each bin (based on bin edges), cuts into a
    numeric feature, assigns each entry to a predefined bin with specified 
    bin edges and labels, and appends assigned bin entries to a new column.

    Parameters
    ----------    
    feature_df : pandas.Dataframe
        Categorical features, their label entries and the corresponding 
        target class.
    feature : str
        Name of feature.
    edges : array of int
        Numeric bin edges.

    Returns
    -------   
    feature_df : pandas.Dataframe
        Feature data with appended column of assigned bin entries.
    """
    labels = [f"{edges[edge] + 1} \u2013 {edges[edge + 1]}" 
              for edge in range(len(edges) - 1)]
    
    feature_df[f"{feature}_binned"] = pd.cut(feature_df[feature], 
                                             bins = edges, 
                                             labels = labels)
    return feature_df    
#---------------------------------------------------------------------------------------
def plot_heatmap(matrix_df, fig_size, c_map, annot_size, bounds, title, 
                 title_size, bar_label, file_name, map_mask = None):
    """
    Produces and saves (to file) a formatted visual of a heatmap displaying
    data with feature-indexed rows and columns with a specified colour 
    scheme.

    Parameters
    ----------
    matrix_df : pandas Dataframe 
        Matrix of data that the heatmap plots.
    fig_size : tuple of int 
        Dimensions of the figure.
    c_map : Colormap 
        Custom colourmap for the heatmap. 
    annot_size : int
        Size of data annotations.
    bounds : tuple of float 
        Minimum and maximum boundaries for colour scale.
    title : str 
        Title of the figure.
    title_size : int
        Size of the figure title. 
    bar_label : str
        Label for the heatmap colourbar. 
    file_name : str 
        Name of visual saved to file.
    map_mask : array of float
        Array for hiding upper half of matrix.
    """
    figure, axes = plt.subplots(figsize = fig_size)

    sns.heatmap(matrix_df,
                cmap = c_map,        
                vmin = bounds[0], 
                vmax = bounds[1],
                fmt = ".2f",         # float with 2 decimal places          
                annot = True,                  
                annot_kws = {"size": annot_size},
                linewidth = 0.4,
                ax = axes,
                mask = map_mask)
    
    # chart formatting:
    axes.tick_params(axis = "both", labelsize = 11)
    axes.set_xlabel("Feature", fontsize = 12)
    axes.set_ylabel("Feature", fontsize = 12)
    axes.set_title(title, fontsize = title_size, fontweight = "bold", pad = 8)
    
    colour_bar = axes.collections[0].colorbar 
    colour_bar.set_label(bar_label, fontsize = 12)
    
    plt.tight_layout()
    plt.savefig(f"reports/figures/{file_name}.png", dpi = 300, bbox_inches = "tight")
    plt.show()
#---------------------------------------------------------------------------------------    