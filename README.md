Comparison-of-Gaussian-and-Student-s-T-Copulas-in-Statistical-Arbitrage.
This code generates various graphs and metrics comparing both copulas, using ideas from my previous projects. More in depth analysis is coming!
Main source: https://fbe.unimelb.edu.au/__data/assets/pdf_file/0003/2592021/146.pdf


INFO:

We use two copulas, Gaussian and Student's T. Within each, we assess a rolling correlation, a fixed correlation and a smoothed correlation with EWMA. 

ANALYSIS:

There are various parameters here. The most notable are:
- The sensitivity of the prices of two stocks.
- The degrees of freedom for the Student's T distribution, thereby effecting the tail dependence.
- The smoothing factor.

In config.py,  we can select a range of these. On a 10y period with some highly correlated and somewhat correlated companies:
-  3 degrees of freedom work best (highest average Sharpe) for T fixed and rolling (which does fit the estimate v from source), but 4 degrees of freedom are best for the filtered model.

  <img width="832" height="499" alt="image" src="https://github.com/user-attachments/assets/e854f36c-8565-4e5a-a79b-0bb6fdc15622" />

-  There is no large difference over smoothing factors, but it seems to influence the Gaussian copula more.
<img width="828" height="498" alt="image" src="https://github.com/user-attachments/assets/0433368f-4981-4a32-9647-2a71cab3f918" />

  
-  Trading signal is interesting, showing best performance at 0.025.
<img width="831" height="497" alt="image" src="https://github.com/user-attachments/assets/4cdbee4f-174e-4d63-9c95-f9f73606ef9a" />
