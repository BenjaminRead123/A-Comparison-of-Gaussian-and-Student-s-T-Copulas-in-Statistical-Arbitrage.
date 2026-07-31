percentSplit = 0.7
time = "10y"
v = 3
signalSensitivity = 0.05
rhoSmoothingAlpha = 0.50
rollingCorrelationLookback = 126

baseParameters = {
    "percentSplit": percentSplit,
    "time": time,
    "v": v,
    "rollingCorrelationLookback": rollingCorrelationLookback,
    "rhoSmoothingAlpha": rhoSmoothingAlpha
}

pairs = [
    # Highly correlated
    ("KO", "PEP", "SPY", "Highly correlated"),
    #("V", "MA", "SPY", "Highly correlated"),
    ("XOM", "CVX", "SPY", "Highly correlated"),
    ("CBA.AX", "NAB.AX", "^AXJO", "Highly correlated"),
    ("BHP.AX", "RIO.AX", "^AXJO", "Highly correlated"),

    # Somewhat correlated
    ("KO", "MNST", "SPY", "Somewhat correlated"),
    ("PEP", "MNST", "SPY", "Somewhat correlated"),
    ("MCD", "SBUX", "SPY", "Somewhat correlated"),
    ("WMT", "COST", "SPY", "Somewhat correlated"),
    ("NKE", "LULU", "SPY", "Somewhat correlated"),
    ("AMD", "NVDA", "SPY", "Somewhat correlated"),

    
]

parameterGrid = {
    "v": [3, 4, 5, 7],
    "rhoSmoothingAlpha": [0.10, 0.50],
    "signalSensitivity": [0.025, 0.05, 0.10]
}
