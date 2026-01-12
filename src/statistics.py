# src/statistics.py
#STATISTICAL TOOLS FOR PAIR TRADING STRATEGY DEVELOPMENT
import pandas as pd
import numpy as np
from numpy import log, polyfit, sqrt, std, subtract
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

#section 1 --> tradability metrics for pairs
def calculate_hurst(ts):
    #calculates the hurst exponant to test for mean reversion
    # H < 0.5 = Mean Reverting (The series wants to return to the average)
    # H = 0.5 = Random Walk (Geometric Brownian Motion - Unpredictable)
    # H > 0.5 = Trending (Momentum)
    lags = range(2, 100)
    # Calculate the standard deviation of differences for various lags
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # Use linear regression to find the scaling exponent
    poly = polyfit(log(lags), log(tau), 1)
    return poly[0] * 2.0

def calculate_tradability_metrics(price_data, stock_a, stock_b):
    """
    Calculates Half-Life, Hurst, and Zero Crossings.
    Includes robust error handling to identify where it fails.
    """
    try:
        # --- 1. GET DATA ---
        s1 = price_data[stock_a]
        s2 = price_data[stock_b]
        
        # Align data (Drop NaNs)
        df = pd.concat([s1, s2], axis=1).dropna()
        if len(df) < 50: # Require at least 50 data points
            return None
        
        s1 = df.iloc[:, 0]
        s2 = df.iloc[:, 1]

        # --- 2. HEDGE RATIO ---
        # We assume spread = s1 - (beta * s2)
        x = sm.add_constant(s2)
        model = sm.OLS(s1, x).fit()
        
        # Robustly get the hedge ratio (slope)
        if len(model.params) < 2:
            return None # Regression failed to find a slope
        
        # Use ILOC to get the 2nd parameter (the slope), ignoring the name
        hedge_ratio = model.params.iloc[1]

        # --- 3. CONSTRUCT SPREAD ---
        spread = s1 - (hedge_ratio * s2)

        # --- 4. HALF-LIFE (The likely crash point) ---
        spread_lag = spread.shift(1)
        spread_ret = spread - spread_lag
        
        # Clean up the lags
        df_ou = pd.DataFrame({'ret': spread_ret, 'lag': spread_lag}).dropna()
        
        # OLS for Mean Reversion
        lag_with_const = sm.add_constant(df_ou['lag'])
        model_ou = sm.OLS(df_ou['ret'], lag_with_const).fit()
        
        # CHECK: Did the model return 2 parameters (Intercept + Slope)?
        if len(model_ou.params) < 2:
            # If standard deviation is 0, the model might drop the constant/slope
            return None
            
        # Get lambda (the slope of mean reversion)
        lambda_param = model_ou.params.iloc[1]
        
        # Calculate Half Life
        if lambda_param >= 0:
            half_life = np.inf # Diverging spread
        else:
            half_life = -np.log(2) / lambda_param

        # --- 5. HURST EXPONENT ---
        hurst = calculate_hurst(spread.values)

        # --- 6. ZERO CROSSINGS ---
        centered_spread = spread - spread.mean()
        # Count how many times the sign changes
        zero_crossings = len(np.where(np.diff(np.sign(centered_spread)))[0])

        return {
            'Hedge_Ratio': hedge_ratio,
            'Half_Life': round(half_life, 2),
            'Hurst_Exponent': round(hurst, 4),
            'Zero_Crossings': zero_crossings
        }

    except Exception as e:
        # Silent failure allows the main loop to continue to the next pair
        # Uncomment the print below if you want to see the specific error in console
        # print(f"DEBUG: Failed on {stock_a}-{stock_b}: {e}")
        return None

#section 2 --> unsupervised learning used in 01_pair_search

def get_clusters(returns_df, n_components=5, eps=0.25):
    #perform PCA to extract latent factors
    #then cluster stocks using DBSCAN

    # n_components (int) : number of PCA components to extract
    #eps (float) : DBSCAN epsilon (distance threshold)

    #returns a datframe with tickers and clusters columns

   #1-PCA
    pca = PCA(n_components=n_components)
    pca.fit(returns_df)

    #get the factor loadings and invert the matrix
    X = pca.components_.T
    #normalize so volatile stocks don't distort distance
    X = Normalizer().fit_transform(X)

    #2 - cluster the factors using DBSCAN
    clf = DBSCAN(eps=eps, min_samples=2)
    labels = clf.fit_predict(X)

    clustered_df = pd.DataFrame({'Ticker':returns_df.columns,
                                 'Cluster': labels})
    
    #filter out the noise
    clean_clusters = clustered_df[clustered_df['Cluster'] != -1].sort_values('Cluster')

    return clean_clusters


def find_cointegrated_pairs(price_data, clusters_df):
    
    # Step 3: The Engle-Granger Test (RELAXED VERSION)
    
    pairs = []
    unique_clusters = clusters_df['Cluster'].unique()
    
    print(f"Testing Cointegration on {len(unique_clusters)} clusters...")
    
    for cluster_id in unique_clusters:
        tickers = clusters_df[clusters_df['Cluster'] == cluster_id]['Ticker'].tolist()
        n = len(tickers)
        
        # --- CHANGE 1: REMOVED THE LIMIT ---
        # We allow clusters up to 30 stocks now.
        if n > 30: 
            print(f"  Skipping Cluster {cluster_id} (Too massive: {n} stocks)")
            continue
            
        print(f"  > Checking Cluster {cluster_id} ({n} stocks)...")

        for i in range(n):
            for j in range(i+1, n):
                # Data Cleaning for the pair
                s1 = price_data[tickers[i]]
                s2 = price_data[tickers[j]]
                df_pair = pd.concat([s1, s2], axis=1).dropna()
                
                if len(df_pair) < 100:
                    continue

                clean_s1 = df_pair.iloc[:, 0]
                clean_s2 = df_pair.iloc[:, 1]
                
                try:
                    score, pvalue, _ = coint(clean_s1, clean_s2)
                    
                    # --- CHANGE 2: RELAXED P-VALUE ---
                    # Changed from 0.05 to 0.10 to find "Good enough" pairs
                    if pvalue < 0.10:
                        pairs.append({
                            'Stock A': tickers[i], 
                            'Stock B': tickers[j], 
                            'P-Value': round(pvalue, 5),
                            'Cluster': cluster_id
                        })
                except:
                    continue
                    
    return pd.DataFrame(pairs)
