Comparison-of-Gaussian-and-Student-s-T-Copulas-in-Statistical-Arbitrage.
This code generates various graphs and metrics comparing both copulas.

INFO:

We use two copulas, Gaussian and Student's T. Within each, we assess a rolling correlation, a fixed correlation and a smoothed correlation with EWMA. There is a look back period required for the rolling correlation, which we choose 63, 126 and 252. 

ANALYSIS:

There are various parameters here. The most notable are:
- The sensitivity of the prices of two stocks.
- The degrees of freedom for the Student's T distribution, thereby effecting the tail dependence.
- The smoothing factor.

In config.py, we can compare these, we can select a range of these. On a 10y period with some highly correlated and somewhat correlated companies:
-  3 degrees of freedom work best (highest average Sharpe) for T fixed and rolling, but 4 degrees of freedom are best for the filtered model.
-  There is no large difference over smoothing factors, but it seems to influence the Gaussian copula more. 
-  Trading signal is interesting, showing best performance at 0.025.
