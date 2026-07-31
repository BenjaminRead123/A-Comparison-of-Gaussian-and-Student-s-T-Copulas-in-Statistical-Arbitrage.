Comparison-of-Gaussian-and-Student-s-T-Copulas-in-Statistical-Arbitrage.
This code generates various graphs and metrics comparing both copulas. Main source: https://fbe.unimelb.edu.au/__data/assets/pdf_file/0003/2592021/146.pdf


INFO:

We use two copulas, Gaussian and Student's T. Within each, we assess a rolling correlation, a fixed correlation and a smoothed correlation with EWMA. 

ANALYSIS:

There are various parameters here. The most notable are:
- The sensitivity of the prices of two stocks.
- The degrees of freedom for the Student's T distribution, thereby effecting the tail dependence.
- The smoothing factor.

In config.py, we can compare these, we can select a range of these. On a 10y period with some highly correlated and somewhat correlated companies:
-  3 degrees of freedom work best (highest average Sharpe) for T fixed and rolling, but 4 degrees of freedom are best for the filtered model.
-  There is no large difference over smoothing factors, but it seems to influence the Gaussian copula more. 
-  Trading signal is interesting, showing best performance at 0.025.
