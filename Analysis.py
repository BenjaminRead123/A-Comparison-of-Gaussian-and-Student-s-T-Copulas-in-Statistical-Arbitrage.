import itertools
import pandas as pd
import matplotlib.pyplot as plt

from main import runBacktest

from config import (
    baseParameters,
    pairs,
    parameterGrid
)

def graphVEffect(results):
    tOnly = results[
        results["strategy"].str.contains("T")
    ]

    summary = (
        tOnly
        .groupby(["v", "strategy"])["sharpe"]
        .mean()
        .unstack()
    )

    summary.plot(
        marker="o",
        figsize=(10, 6)
    )

    plt.title("Effect of T degrees of freedom")
    plt.xlabel("v")
    plt.ylabel("Average Sharpe")
    plt.tight_layout()
    plt.savefig("v_effect.png", dpi=200)
    plt.show()


def graphAlphaEffect(results):
    filteredOnly = results[
        results["strategy"].isin([
            "T filtered rolling",
            "Gaussian filtered rolling"
        ])
    ]

    summary = (
        filteredOnly
        .groupby(["rhoSmoothingAlpha", "strategy"])["sharpe"]
        .mean()
        .unstack()
    )

    summary.plot(
        marker="o",
        figsize=(10, 6)
    )

    plt.title("Effect of rho smoothing alpha")
    plt.xlabel("rhoSmoothingAlpha")
    plt.ylabel("Average Sharpe")
    plt.tight_layout()
    plt.savefig("alpha_effect.png", dpi=200)
    plt.show()


def graphSensitivityEffect(results):
    summary = (
        results
        .groupby(["signalSensitivity", "strategy"])["sharpe"]
        .mean()
        .unstack()
    )

    summary.plot(
        marker="o",
        figsize=(10, 6)
    )

    plt.title("Effect of trading signal sensitivity")
    plt.xlabel("signalSensitivity")
    plt.ylabel("Average Sharpe")
    plt.tight_layout()
    plt.savefig("sensitivity_effect.png", dpi=200)
    plt.show()
def generateParameterCombinations():
    keys = list(parameterGrid.keys())
    values = list(parameterGrid.values())

    for combination in itertools.product(*values):
        parameters = baseParameters.copy()

        for key, value in zip(keys, combination):
            parameters[key] = value

        yield parameters


def runAnalysis():
    allRows = []

    parameterCombinations = list(
        generateParameterCombinations()
    )

    totalRuns = len(pairs) * len(parameterCombinations)
    runNumber = 1

    for pair in pairs:
        ticker1, ticker2, market, group = pair

        for parameters in parameterCombinations:
            print(
        f"[{runNumber}/{totalRuns}] "
        f"Testing {ticker1}/{ticker2} | "
        f"time={parameters.get('time')} | "
        f"rolling={parameters.get('rollingCorrelationLookback')} | "
        f"v={parameters.get('v')} | "
        f"alpha={parameters.get('rhoSmoothingAlpha')} | "
        f"sensitivity={parameters.get('signalSensitivity')}"
) 

            try:
                rows = runBacktest(
                    ticker1,
                    ticker2,
                    market,
                    parameters
                )

                for row in rows:
                    row["group"] = group

                allRows.extend(rows)

            except Exception as error:
                print(
                    "FAILED:",
                    ticker1,
                    ticker2,
                    error
                )

            runNumber += 1

    results = pd.DataFrame(allRows)

    results.to_csv(
        "correlation_experiment_results.csv",
        index=False
    )

    return results


def graphAverageSharpeByStrategy(results):
    summary = (
        results
        .groupby("strategy")["sharpe"]
        .mean()
        .sort_values()
    )

    summary.plot(
        kind="barh",
        figsize=(10, 6)
    )

    plt.title("Average Sharpe by strategy")
    plt.xlabel("Average Sharpe")
    plt.ylabel("Strategy")
    plt.tight_layout()
    plt.savefig(
        "average_sharpe_by_strategy.png",
        dpi=200
    )
    plt.show()


def graphStrategyByPair(results):
    summary = (
        results
        .groupby(["pair", "strategy"])["sharpe"]
        .mean()
        .unstack()
    )

    summary.plot(
        kind="bar",
        figsize=(14, 7)
    )

    plt.title("Average Sharpe by pair and strategy")
    plt.xlabel("Pair")
    plt.ylabel("Average Sharpe")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        "sharpe_by_pair_and_strategy.png",
        dpi=200
    )
    plt.show()


def graphStrategyByCorrelationGroup(results):
    summary = (
        results
        .groupby(["group", "strategy"])["sharpe"]
        .mean()
        .unstack()
    )

    summary.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Average Sharpe by pair type")
    plt.xlabel("Pair group")
    plt.ylabel("Average Sharpe")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        "strategy_by_pair_type.png",
        dpi=200
    )
    plt.show()


def graphRollingLookbackEffect(results):
    rollingOnly = results[
        results["strategy"].isin([
            "T rolling",
            "Gaussian rolling"
        ])
    ]

    summary = (
        rollingOnly
        .groupby([
            "rollingCorrelationLookback",
            "strategy"
        ])["sharpe"]
        .mean()
        .unstack()
    )

    summary.plot(
        marker="o",
        figsize=(10, 6)
    )

    plt.title("Effect of rolling correlation lookback")
    plt.xlabel("Rolling correlation lookback")
    plt.ylabel("Average Sharpe")
    plt.tight_layout()
    plt.savefig(
        "rolling_lookback_effect.png",
        dpi=200
    )
    plt.show()


def graphFixedVsRollingT(results):
    tOnly = results[
        results["strategy"].isin([
            "T fixed",
            "T rolling"
        ])
    ]

    summary = (
        tOnly
        .groupby(["pair", "strategy"])["sharpe"]
        .mean()
        .unstack()
    )

    summary["T rolling minus T fixed"] = (
        summary["T rolling"]
        - summary["T fixed"]
    )

    summary["T rolling minus T fixed"].sort_values().plot(
        kind="barh",
        figsize=(10, 6)
    )

    plt.axvline(0, linestyle="--")
    plt.title("T rolling versus T fixed by pair")
    plt.xlabel("Sharpe difference")
    plt.ylabel("Pair")
    plt.tight_layout()
    plt.savefig(
        "t_rolling_minus_t_fixed.png",
        dpi=200
    )
    plt.show()


def graphFixedVsRollingGaussian(results):
    gaussianOnly = results[
        results["strategy"].isin([
            "Gaussian fixed",
            "Gaussian rolling"
        ])
    ]

    summary = (
        gaussianOnly
        .groupby(["pair", "strategy"])["sharpe"]
        .mean()
        .unstack()
    )

    summary["Gaussian rolling minus Gaussian fixed"] = (
        summary["Gaussian rolling"]
        - summary["Gaussian fixed"]
    )

    summary[
        "Gaussian rolling minus Gaussian fixed"
    ].sort_values().plot(
        kind="barh",
        figsize=(10, 6)
    )

    plt.axvline(0, linestyle="--")
    plt.title("Gaussian rolling versus Gaussian fixed by pair")
    plt.xlabel("Sharpe difference")
    plt.ylabel("Pair")
    plt.tight_layout()
    plt.savefig(
        "gaussian_rolling_minus_gaussian_fixed.png",
        dpi=200
    )
    plt.show()


def printBestStrategyByGroup(results):
    summary = (
        results
        .groupby(["group", "strategy"])["sharpe"]
        .mean()
        .reset_index()
        .sort_values(
            ["group", "sharpe"],
            ascending=[True, False]
        )
    )

    print()
    print("=" * 80)
    print("AVERAGE SHARPE BY GROUP")
    print("=" * 80)
    print(summary.to_string(index=False))

    winners = (
        summary
        .sort_values("sharpe", ascending=False)
        .groupby("group")
        .head(1)
    )

    print()
    print("=" * 80)
    print("BEST STRATEGY BY GROUP")
    print("=" * 80)
    print(winners.to_string(index=False))


def printBestStrategyByPair(results):
    summary = (
        results
        .groupby(["pair", "strategy"])["sharpe"]
        .mean()
        .reset_index()
    )

    winners = (
        summary
        .sort_values("sharpe", ascending=False)
        .groupby("pair")
        .head(1)
    )

    print()
    print("=" * 80)
    print("BEST STRATEGY BY PAIR")
    print("=" * 80)
    print(winners.to_string(index=False))


def printBestConfigurations(results):
    columns = [
        "pair",
        "group",
        "strategy",
        "time",
        "rollingCorrelationLookback",
        "totalReturn",
        "annualisedVolatility",
        "sharpe",
        "maxDrawdown"
    ]

    best = (
        results
        .sort_values("sharpe", ascending=False)
        .head(25)
    )

    print()
    print("=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)
    print(best[columns].to_string(index=False))


def main():
    results = runAnalysis()

    printBestConfigurations(results)
    printBestStrategyByGroup(results)
    printBestStrategyByPair(results)

    graphAverageSharpeByStrategy(results)
    graphStrategyByPair(results)
    graphStrategyByCorrelationGroup(results)
    graphRollingLookbackEffect(results)
    graphFixedVsRollingT(results)
    graphFixedVsRollingGaussian(results)
    graphVEffect(results)
    graphAlphaEffect(results)
    graphSensitivityEffect(results)

if __name__ == "__main__":
    main()
