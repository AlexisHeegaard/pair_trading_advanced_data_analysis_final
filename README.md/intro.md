#Statistical Arbitrage : A Pair Trading Framework

##Project Overview
The aim of this project is to test and optimize a pair trading strategy using Machine learning algorithms on S&P 500 daily data segmented by GICS sector 

##Methodology & Statistical Reasoning
###Data Preprocessing
**liquiditity filter** To ensure tradability, we discard the bottom 25% of stocks by average daily volume within each sector

**data integrity** Assets with >10% missing data are removed, while remaining gaps are forward-filled

###Dimensionality reduction & Clustering
**PCA** we apply principal component analysis to extract 5 latent factors driving market movements, filtering out idiosyncratic noise
**DBSCAN** density-based spatial clustering identifies "natural" clusters of assets without forcing noise point into a group

### dynamic state-space Modelling
**klaman filter** to overcome the "ghosting effect" of fixed lookback windows, we employ a linear kalman filter to estimate the hidden hedge ratio recursively


