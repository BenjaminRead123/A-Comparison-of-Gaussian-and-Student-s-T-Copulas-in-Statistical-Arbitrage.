import itertools
import pandas as pd
import matplotlib.pyplot as plt

from main import runBacktest

from config import (
    baseParameters,
    pairs,
    parameterGrid
)


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


def graphVEffect(results):
    if "v" not in results.columns:
        print("No v column found.")
        return

    tOnly = results[
        results["strategy"].str.contains("T")
    ]

    if tOnly.empty:
        print("No T strategies found.")
        return

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
    if "rhoSmoothingAlpha" not in results.columns:
        print("No rhoSmoothingAlpha column found.")
        return

    filteredOnly = results[
        results["strategy"].isin([
            "T filtered rolling",
            "Gaussian filtered rolling"
        ])
    ]

    if filteredOnly.empty:
        print("No filtered rolling strategies found.")
        return

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
    if "signalSensitivity" not in results.columns:
        print("No signalSensitivity column found.")
        return

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
    if "rollingCorrelationLookback" not in results.columns:
        print("No rollingCorrelationLookback column found.")
        return

    rollingOnly = results[
        results["strategy"].isin([
            "T rolling",
            "Gaussian rolling",
            "T filtered rolling",
            "Gaussian filtered rolling"
        ])
    ]

    if rollingOnly.empty:
        print("No rolling strategies found.")
        return

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
            "T rolling",
            "T filtered rolling"
        ])
    ]

    if tOnly.empty:
        print("No T strategies found.")
        return

    summary = (
        tOnly
        .groupby(["pair", "strategy"])["sharpe"]
        .mean()
        .unstack()
    )

    summary.plot(
        kind="bar",
        figsize=(14, 7)
    )

    plt.title("T strategies by pair")
    plt.xlabel("Pair")
    plt.ylabel("Average Sharpe")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        "t_strategies_by_pair.png",
        dpi=200
    )
    plt.show()


def graphFixedVsRollingGaussian(results):
    gaussianOnly = results[
        results["strategy"].isin([
            "Gaussian fixed",
            "Gaussian rolling",
            "Gaussian filtered rolling"
        ])
    ]

    if gaussianOnly.empty:
        print("No Gaussian strategies found.")
        return

    summary = (
        gaussianOnly
        .groupby(["pair", "strategy"])["sharpe"]
        .mean()
        .unstack()
    )

    summary.plot(
        kind="bar",
        figsize=(14, 7)
    )

    plt.title("Gaussian strategies by pair")
    plt.xlabel("Pair")
    plt.ylabel("Average Sharpe")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        "gaussian_strategies_by_pair.png",
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
        "v",
        "rhoSmoothingAlpha",
        "signalSensitivity",
        "totalReturn",
        "annualisedVolatility",
        "sharpe",
        "maxDrawdown"
    ]

    existingColumns = [
        column for column in columns
        if column in results.columns
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
    print(best[existingColumns].to_string(index=False))


PLOT_OPTIONS = {
    "1": {
        "name": "Average Sharpe by strategy",
        "function": graphAverageSharpeByStrategy
    },
    "2": {
        "name": "Sharpe by pair and strategy",
        "function": graphStrategyByPair
    },
    "3": {
        "name": "Sharpe by correlation group",
        "function": graphStrategyByCorrelationGroup
    },
    "4": {
        "name": "Rolling lookback effect",
        "function": graphRollingLookbackEffect
    },
    "5": {
        "name": "T strategies by pair",
        "function": graphFixedVsRollingT
    },
    "6": {
        "name": "Gaussian strategies by pair",
        "function": graphFixedVsRollingGaussian
    },
    "7": {
        "name": "Effect of T degrees of freedom v",
        "function": graphVEffect
    },
    "8": {
        "name": "Effect of smoothing alpha",
        "function": graphAlphaEffect
    },
    "9": {
        "name": "Effect of signal sensitivity",
        "function": graphSensitivityEffect
    }
}


def choosePlots(results):
    print()
    print("=" * 80)
    print("CHOOSE GRAPHS TO PLOT")
    print("=" * 80)

    for key, option in PLOT_OPTIONS.items():
        print(f"{key}. {option['name']}")

    print()
    print("Type graph numbers separated by commas.")
    print("Example: 1,2,7")
    print("Type 'all' to plot everything.")
    print("Type 'none' to skip graphs.")
    print()

    choice = input("Graphs to plot: ").strip().lower()

    if choice == "none":
        print("Skipping graphs.")
        return

    if choice == "all":
        selectedKeys = list(PLOT_OPTIONS.keys())
    else:
        selectedKeys = [
            key.strip()
            for key in choice.split(",")
            if key.strip() in PLOT_OPTIONS
        ]

    if len(selectedKeys) == 0:
        print("No valid graph choices selected.")
        return

    for key in selectedKeys:
        option = PLOT_OPTIONS[key]

        print()
        print(f"Plotting: {option['name']}")

        try:
            option["function"](results)

        except Exception as error:
            print(
                "FAILED TO PLOT:",
                option["name"],
                "|",
                error
            )


def choosePrints(results):
    print()
    print("=" * 80)
    print("CHOOSE TABLES TO PRINT")
    print("=" * 80)

    print("1. Best configurations")
    print("2. Best strategy by group")
    print("3. Best strategy by pair")
    print("4. All")
    print("5. None")

    choice = input("Tables to print: ").strip().lower()

    if choice == "1":
        printBestConfigurations(results)

    elif choice == "2":
        printBestStrategyByGroup(results)

    elif choice == "3":
        printBestStrategyByPair(results)

    elif choice == "4" or choice == "all":
        printBestConfigurations(results)
        printBestStrategyByGroup(results)
        printBestStrategyByPair(results)

    elif choice == "5" or choice == "none":
        print("Skipping printed tables.")

    else:
        print("Invalid choice, skipping printed tables.")


def main():
    results = runAnalysis()

    choosePrints(results)
    choosePlots(results)


if __name__ == "__main__":
    main()
