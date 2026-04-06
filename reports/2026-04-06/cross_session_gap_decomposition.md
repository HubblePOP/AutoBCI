# cross-session gap decomposition

## ridge family

- `cross-session test r`：`0.2322`
- `upper-bound test r`：`0.2953`
- `Δr`：`0.0631`
- `ΔMAE`：`0.0222`
- `ΔRMSE`：`0.4102`
- `Δgain`：`0.1315`
- 最拖后腿的 3 个关节：Mcp(+0.2966), Elb(+0.2402), Kne(+0.2361)

## feature-LSTM family

- `cross-session test r`：`0.3483`
- `upper-bound test r`：`0.4745`
- `Δr`：`0.1262`
- `ΔMAE`：`-1.0076`
- `ΔRMSE`：`-0.7883`
- `Δgain`：`0.0732`
- 最拖后腿的 3 个关节：Kne(+0.1563), Mcp(+0.1396), Sho(+0.1172)

## XGBoost family

- `cross-session test r`：`0.3712`
- `upper-bound test r`：`0.4723`
- `Δr`：`0.1010`
- `ΔMAE`：`-0.5294`
- `ΔRMSE`：`-0.6720`
- `Δgain`：`-0.0277`
- 最拖后腿的 3 个关节：Kne(+0.0544), Mcp(+0.0496), Elb(+0.0282)
