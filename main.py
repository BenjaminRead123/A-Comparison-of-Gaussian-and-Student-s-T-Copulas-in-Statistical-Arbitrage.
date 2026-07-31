
import pandas as pd
import numpy as np
import yfinance as yf
import scipy
import warnings

from config import (
    percentSplit,
    rollingCorrelationLookback,
    time,
    v,
    signalSensitivity,
    rhoSmoothingAlpha,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


returnsArrayT = []
returnsArrayTUpdated = []
returnsArrayGaussian = []
returnsArrayGaussianUpdated = []
returnsArrayTFiltered = []
returnsArrayGaussianFiltered = []

cba = yf.Ticker("KO")
nab = yf.Ticker("PEP")


def trainingData(data, df=None):
    if df is None:
        df = v

    priceData = data.history(
        period=time,
        auto_adjust=True
    )

    priceData = priceData.dropna()

    splitPoint = int(len(priceData) * percentSplit)

    trainingData = priceData.iloc[:splitPoint].copy()
    realData = priceData.iloc[splitPoint:].copy()

    trainingData.loc[:, "Log Returns"] = np.log(
        trainingData.loc[:, "Close"]
        / trainingData.loc[:, "Close"].shift(1)
    )

    trainingData = trainingData.dropna()

    n = trainingData["Log Returns"].count()

    trainingData.loc[:, "Rank"] = (
        trainingData["Log Returns"]
        .rank(method="average")
    )

    trainingData.loc[:, "Percentile"] = (
        trainingData.loc[:, "Rank"] / (n + 1)
    )

    trainingData.loc[:, "Normal Score"] = (
        scipy.stats.norm.ppf(
            trainingData.loc[:, "Percentile"]
        )
    )

    trainingData.loc[:, "T Score"] = (
        scipy.stats.t.ppf(
            trainingData.loc[:, "Percentile"],
            df
        )
    )

    realData.loc[:, "Log Returns"] = np.log(
        realData.loc[:, "Close"]
        / realData.loc[:, "Close"].shift(1)
    )

    realData = realData.dropna()

    sortedTrainingReturns = np.sort(
        trainingData.loc[:, "Log Returns"]
    )

    n = len(sortedTrainingReturns)

    percentiles = []

    for r in realData.loc[:, "Log Returns"]:
        count = np.searchsorted(
            sortedTrainingReturns,
            r,
            side="right"
        )

        percentile = (count + 1) / (n + 2)

        percentiles.append(percentile)

    realData.loc[:, "Percentile"] = percentiles

    realData.loc[:, "Normal Score"] = (
        scipy.stats.norm.ppf(
            realData.loc[:, "Percentile"]
        )
    )

    realData.loc[:, "T Score"] = (
        scipy.stats.t.ppf(
            realData.loc[:, "Percentile"],
            df
        )
    )

    return trainingData, realData


trainingCBA, realCBA = trainingData(cba)
trainingNAB, realNAB = trainingData(nab)


def estimateV():
    arr = []
    lowest = 0

    for y in range(3, 10):
        trainingCBAForV, _ = trainingData(cba, y)
        trainingNABForV, _ = trainingData(nab, y)

        data1 = trainingCBAForV
        data2 = trainingNABForV

        rho = TrainT(data1, data2)

        R = np.array([
            [1, rho],
            [rho, 1]
        ])

        tx = data1["T Score"]
        ty = data2["T Score"]

        SSE = 0

        for i in range(tx.count()):
            Tcopi = scipy.stats.multivariate_t.cdf(
                [tx.iloc[i], ty.iloc[i]],
                shape=R,
                df=y
            )

            empiricalProb = (
                (
                    (tx <= tx.iloc[i])
                    & (ty <= ty.iloc[i])
                ).sum()
                / tx.count()
            )

            SSE += np.pow(Tcopi - empiricalProb, 2)

        arr.append(SSE)

        if SSE < arr[lowest]:
            lowest = y - 3

    print(
        "Lowest is",
        arr[lowest],
        "with",
        lowest + 3,
        "degrees of freedom."
    )


def Train(data1, data2):
    zx = data1["Normal Score"]
    zy = data2["Normal Score"]

    rho = zx.corr(zy)

    return rho


def TrainT(data1, data2):
    tx = data1["T Score"]
    ty = data2["T Score"]

    tau = tx.corr(
        ty,
        method="kendall"
    )

    rho = np.sin(np.pi * tau / 2)

    return rho


def safeRho(rho, fallback):
    if pd.isna(rho):
        rho = fallback

    return float(np.clip(rho, -0.99, 0.99))


def getRollingGaussianRho(
    fullX,
    fullY,
    fullPosition,
    lookback,
    fallback
):
    start = max(
        0,
        fullPosition - lookback
    )

    xWindow = fullX.iloc[start:fullPosition]
    yWindow = fullY.iloc[start:fullPosition]

    rho = xWindow.corr(yWindow)

    return safeRho(rho, fallback)


def getRollingTRho(
    fullX,
    fullY,
    fullPosition,
    lookback,
    fallback
):
    start = max(
        0,
        fullPosition - lookback
    )

    xWindow = fullX.iloc[start:fullPosition]
    yWindow = fullY.iloc[start:fullPosition]

    tau = xWindow.corr(
        yWindow,
        method="kendall"
    )

    if pd.isna(tau):
        return fallback

    rho = np.sin(np.pi * tau / 2)

    return safeRho(rho, fallback)


def checkHx(hx, i, position):
    if position[i - 1] == 0:
        if hx[i] <= signalSensitivity:
            position.append(1)
        elif hx[i] >= 1-signalSensitivity:
            position.append(-1)
        else:
            position.append(0)

    elif position[i - 1] == 1:
        if hx[i] > 0.55:
            position.append(0)
        else:
            position.append(1)

    elif position[i - 1] == -1:
        if hx[i] < 0.45:
            position.append(0)
        else:
            position.append(-1)


def StrategyT():
    position = [0]

    rho = TrainT(
        trainingCBA,
        trainingNAB
    )

    totalReturns = 1

    tx = realCBA["T Score"]
    ty = realNAB["T Score"]

    hx = []

    for i in range(ty.count()):
        numerator = tx.iloc[i] - rho * ty.iloc[i]

        denominator = np.sqrt(
            ((v + ty.iloc[i] ** 2) / (v + 1))
            * (1 - rho ** 2)
        )

        hxValue = scipy.stats.t.cdf(
            numerator / denominator,
            v + 1
        )

        hx.append(hxValue)

    for i in range(1, len(hx)):
        checkHx(hx, i, position)

    for i in range(1, len(hx)):
        nabReturn = (
            realNAB["Close"].iloc[i]
            / realNAB["Close"].iloc[i - 1]
            - 1
        )

        cbaReturn = (
            realCBA["Close"].iloc[i]
            / realCBA["Close"].iloc[i - 1]
            - 1
        )

        r = position[i - 1] * (
            cbaReturn - nabReturn
        )

        returnsArrayT.append(r)

        totalReturns *= 1 + r

    return totalReturns - 1


def strategyWithRollingCorrelationT():
    position = [0]

    fallbackRho = TrainT(
        trainingCBA,
        trainingNAB
    )

    totalReturns = 1

    tx = realCBA["T Score"]
    ty = realNAB["T Score"]

    fullTx = pd.concat([
        trainingCBA["T Score"],
        realCBA["T Score"]
    ])

    fullTy = pd.concat([
        trainingNAB["T Score"],
        realNAB["T Score"]
    ])

    hx = []

    for i in range(ty.count()):
        fullPosition = len(trainingCBA) + i

        rho = getRollingTRho(
            fullTx,
            fullTy,
            fullPosition,
            rollingCorrelationLookback,
            fallbackRho
        )

        numerator = tx.iloc[i] - rho * ty.iloc[i]

        denominator = np.sqrt(
            ((v + ty.iloc[i] ** 2) / (v + 1))
            * (1 - rho ** 2)
        )

        hxValue = scipy.stats.t.cdf(
            numerator / denominator,
            v + 1
        )

        hx.append(hxValue)

    for i in range(1, len(hx)):
        checkHx(hx, i, position)

    for i in range(1, len(hx)):
        nabReturn = (
            realNAB["Close"].iloc[i]
            / realNAB["Close"].iloc[i - 1]
            - 1
        )

        cbaReturn = (
            realCBA["Close"].iloc[i]
            / realCBA["Close"].iloc[i - 1]
            - 1
        )

        r = position[i - 1] * (
            cbaReturn - nabReturn
        )

        returnsArrayTUpdated.append(r)

        totalReturns *= 1 + r

    return totalReturns - 1


def StrategyGaussian():
    position = [0]

    rho = Train(
        trainingCBA,
        trainingNAB
    )

    totalReturns = 1

    zx = realCBA["Normal Score"]
    zy = realNAB["Normal Score"]

    hx = scipy.stats.norm.cdf(
        (zx - rho * zy)
        / np.sqrt(1 - rho ** 2)
    )

    for i in range(1, len(hx)):
        checkHx(hx, i, position)

    for i in range(1, len(hx)):
        nabReturn = (
            realNAB["Close"].iloc[i]
            / realNAB["Close"].iloc[i - 1]
            - 1
        )

        cbaReturn = (
            realCBA["Close"].iloc[i]
            / realCBA["Close"].iloc[i - 1]
            - 1
        )

        r = position[i - 1] * (
            cbaReturn - nabReturn
        )

        returnsArrayGaussian.append(r)

        totalReturns *= 1 + r

    return totalReturns - 1


def strategyWithRollingCorrelationGauss():
    position = [0]

    fallbackRho = Train(
        trainingCBA,
        trainingNAB
    )

    totalReturns = 1

    zx = realCBA["Normal Score"]
    zy = realNAB["Normal Score"]

    fullZx = pd.concat([
        trainingCBA["Normal Score"],
        realCBA["Normal Score"]
    ])

    fullZy = pd.concat([
        trainingNAB["Normal Score"],
        realNAB["Normal Score"]
    ])

    hx = []

    for i in range(zy.count()):
        fullPosition = len(trainingCBA) + i

        rho = getRollingGaussianRho(
            fullZx,
            fullZy,
            fullPosition,
            rollingCorrelationLookback,
            fallbackRho
        )

        hxValue = scipy.stats.norm.cdf(
            (zx.iloc[i] - rho * zy.iloc[i])
            / np.sqrt(1 - rho ** 2)
        )

        hx.append(hxValue)

    for i in range(1, len(hx)):
        checkHx(hx, i, position)

    for i in range(1, len(hx)):
        nabReturn = (
            realNAB["Close"].iloc[i]
            / realNAB["Close"].iloc[i - 1]
            - 1
        )

        cbaReturn = (
            realCBA["Close"].iloc[i]
            / realCBA["Close"].iloc[i - 1]
            - 1
        )

        r = position[i - 1] * (
            cbaReturn - nabReturn
        )

        returnsArrayGaussianUpdated.append(r)

        totalReturns *= 1 + r

    return totalReturns - 1


def resetReturnArrays():
    returnsArrayT.clear()
    returnsArrayTUpdated.clear()
    returnsArrayGaussian.clear()
    returnsArrayGaussianUpdated.clear()
    returnsArrayTFiltered.clear()
    returnsArrayGaussianFiltered.clear()

def strategyWithFilteredRollingCorrelationT():
    position = [0]

    fallbackRho = TrainT(
        trainingCBA,
        trainingNAB
    )

    filteredRho = fallbackRho

    totalReturns = 1

    tx = realCBA["T Score"]
    ty = realNAB["T Score"]

    fullTx = pd.concat([
        trainingCBA["T Score"],
        realCBA["T Score"]
    ])

    fullTy = pd.concat([
        trainingNAB["T Score"],
        realNAB["T Score"]
    ])

    hx = []

    for i in range(ty.count()):
        fullPosition = len(trainingCBA) + i

        rollingRho = getRollingTRho(
            fullTx,
            fullTy,
            fullPosition,
            rollingCorrelationLookback,
            fallbackRho
        )

        filteredRho = smoothRho(
            filteredRho,
            rollingRho,
            rhoSmoothingAlpha,
            fallbackRho
        )

        numerator = tx.iloc[i] - filteredRho * ty.iloc[i]

        denominator = np.sqrt(
            ((v + ty.iloc[i] ** 2) / (v + 1))
            * (1 - filteredRho ** 2)
        )

        hxValue = scipy.stats.t.cdf(
            numerator / denominator,
            v + 1
        )

        hx.append(hxValue)

    for i in range(1, len(hx)):
        checkHx(hx, i, position)

    for i in range(1, len(hx)):
        nabReturn = (
            realNAB["Close"].iloc[i]
            / realNAB["Close"].iloc[i - 1]
            - 1
        )

        cbaReturn = (
            realCBA["Close"].iloc[i]
            / realCBA["Close"].iloc[i - 1]
            - 1
        )

        r = position[i - 1] * (
            cbaReturn - nabReturn
        )

        returnsArrayTFiltered.append(r)

        totalReturns *= 1 + r

    return totalReturns - 1

def strategyWithFilteredRollingCorrelationGauss():
    position = [0]

    fallbackRho = Train(
        trainingCBA,
        trainingNAB
    )

    filteredRho = fallbackRho

    totalReturns = 1

    zx = realCBA["Normal Score"]
    zy = realNAB["Normal Score"]

    fullZx = pd.concat([
        trainingCBA["Normal Score"],
        realCBA["Normal Score"]
    ])

    fullZy = pd.concat([
        trainingNAB["Normal Score"],
        realNAB["Normal Score"]
    ])

    hx = []

    for i in range(zy.count()):
        fullPosition = len(trainingCBA) + i

        rollingRho = getRollingGaussianRho(
            fullZx,
            fullZy,
            fullPosition,
            rollingCorrelationLookback,
            fallbackRho
        )

        filteredRho = smoothRho(
            filteredRho,
            rollingRho,
            rhoSmoothingAlpha,
            fallbackRho
        )

        hxValue = scipy.stats.norm.cdf(
            (zx.iloc[i] - filteredRho * zy.iloc[i])
            / np.sqrt(1 - filteredRho ** 2)
        )

        hx.append(hxValue)

    for i in range(1, len(hx)):
        checkHx(hx, i, position)

    for i in range(1, len(hx)):
        nabReturn = (
            realNAB["Close"].iloc[i]
            / realNAB["Close"].iloc[i - 1]
            - 1
        )

        cbaReturn = (
            realCBA["Close"].iloc[i]
            / realCBA["Close"].iloc[i - 1]
            - 1
        )

        r = position[i - 1] * (
            cbaReturn - nabReturn
        )

        returnsArrayGaussianFiltered.append(r)

        totalReturns *= 1 + r

    return totalReturns - 1

def smoothRho(previousFilteredRho, newRollingRho, alpha, fallback):
    if pd.isna(newRollingRho):
        newRollingRho = fallback

    if pd.isna(previousFilteredRho):
        previousFilteredRho = fallback

    filteredRho = (
        alpha * newRollingRho
        + (1 - alpha) * previousFilteredRho
    )

    return safeRho(filteredRho, fallback)


def setupExperiment(
    ticker1,
    ticker2,
    marketTicker,
    parameters
):
    global rhoSmoothingAlpha
    global signalSensitivity
    global cba, nab
    global trainingCBA, realCBA
    global trainingNAB, realNAB
    global percentSplit
    global rollingCorrelationLookback
    global time, v
    global rhoSmoothingAlpha
    rhoSmoothingAlpha = parameters["rhoSmoothingAlpha"]
    signalSensitivity = parameters["signalSensitivity"]
    percentSplit = parameters["percentSplit"]
    rollingCorrelationLookback = parameters[
        "rollingCorrelationLookback"
    ]

    time = parameters["time"]
    v = parameters["v"]

    cba = yf.Ticker(ticker1)
    nab = yf.Ticker(ticker2)

    resetReturnArrays()

    trainingCBA, realCBA = trainingData(cba)
    trainingNAB, realNAB = trainingData(nab)


def getReturnMetrics(returnsArray, totalReturn):
    returns = np.array(
        returnsArray,
        dtype=float
    )

    if len(returns) == 0:
        return {
            "totalReturn": np.nan,
            "annualisedVolatility": np.nan,
            "sharpe": np.nan,
            "maxDrawdown": np.nan
        }

    annualisedVolatility = (
        np.std(returns)
        * np.sqrt(252)
    )

    if np.std(returns) == 0:
        sharpe = np.nan
    else:
        sharpe = (
            np.mean(returns)
            * np.sqrt(252)
            / np.std(returns)
        )

    equityCurve = np.cumprod(1 + returns)

    runningMax = np.maximum.accumulate(
        equityCurve
    )

    drawdown = equityCurve / runningMax - 1

    maxDrawdown = np.min(drawdown)

    return {
        "totalReturn": totalReturn,
        "annualisedVolatility": annualisedVolatility,
        "sharpe": sharpe,
        "maxDrawdown": maxDrawdown
    }


def runBacktest(
    ticker1,
    ticker2,
    marketTicker,
    parameters
):
    setupExperiment(
        ticker1,
        ticker2,
        marketTicker,
        parameters
    )

    rows = []

    resetReturnArrays()
    totalReturn = StrategyT()
    metrics = getReturnMetrics(
        returnsArrayT,
        totalReturn
    )

    rows.append({
        "strategy": "T fixed",
        **metrics
    })

    resetReturnArrays()
    totalReturn = StrategyGaussian()
    metrics = getReturnMetrics(
        returnsArrayGaussian,
        totalReturn
    )

    rows.append({
        "strategy": "Gaussian fixed",
        **metrics
    })

    resetReturnArrays()
    totalReturn = strategyWithRollingCorrelationT()
    metrics = getReturnMetrics(
        returnsArrayTUpdated,
        totalReturn
    )

    rows.append({
        "strategy": "T rolling",
        **metrics
    })
    resetReturnArrays()
    totalReturn = strategyWithFilteredRollingCorrelationT()
    metrics = getReturnMetrics(
        returnsArrayTFiltered,
        totalReturn
    )

    rows.append({
        "strategy": "T filtered rolling",
        **metrics
    })
    resetReturnArrays()
    totalReturn = strategyWithRollingCorrelationGauss()
    metrics = getReturnMetrics(
        returnsArrayGaussianUpdated,
        totalReturn
    )

    rows.append({
        "strategy": "Gaussian rolling",
        **metrics
    })
    resetReturnArrays()
    totalReturn = strategyWithFilteredRollingCorrelationGauss()
    metrics = getReturnMetrics(
        returnsArrayGaussianFiltered,
        totalReturn
    )

    rows.append({
        "strategy": "Gaussian filtered rolling",
        **metrics
    })
    for row in rows:
        row["ticker1"] = ticker1
        row["ticker2"] = ticker2
        row["pair"] = f"{ticker1}/{ticker2}"
        row["market"] = marketTicker

        for key, value in parameters.items():
            row[key] = value

    return rows


def computeTailDependenceT(rho, degreesFreedom):
    if pd.isna(rho):
        return np.nan

    rho = float(
        np.clip(rho, -0.999999, 0.999999)
    )

    if degreesFreedom <= 0:
        raise ValueError(
            "degreesFreedom must be positive"
        )

    argument = np.sqrt(
        (degreesFreedom + 1)
        * (1 - rho)
        / (1 + rho)
    )

    tailDependence = 2 * (
        1
        - scipy.stats.t.cdf(
            argument,
            degreesFreedom + 1
        )
    )

    return float(tailDependence)
def main():
    parameters = {
        "percentSplit": percentSplit,
        "time": time,
        "v": v,
        "rollingCorrelationLookback":
            rollingCorrelationLookback
    }

    rows = runBacktest(
        "KO",
        "PEP",
        "SPY",
        parameters
    )

    results = pd.DataFrame(rows)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
