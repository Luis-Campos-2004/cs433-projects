| stage | preprocessing | model | features | F1 (mean ± std) | accuracy | F1 @ 0.5 | threshold |
|---|---|---|---|---|---|---|---|
| models | BRFSS | least squares | 124 | 0.4089 ± 0.0019 | 0.8557 | 0.0796 | 0.201 |
| models | BRFSS | ridge, lambda=1e-05 | 124 | 0.4089 ± 0.0019 | 0.8557 | 0.0796 | 0.201 |
| models | BRFSS | ridge, lambda=0.0001 | 124 | 0.4089 ± 0.0019 | 0.8563 | 0.0796 | 0.202 |
| models | BRFSS | ridge, lambda=0.001 | 124 | 0.4088 ± 0.0023 | 0.8552 | 0.0789 | 0.200 |
| models | BRFSS | ridge, lambda=0.01 | 124 | 0.4090 ± 0.0033 | 0.8599 | 0.0732 | 0.204 |
| models | BRFSS | logistic | 124 | 0.4230 ± 0.0041 | 0.8696 | 0.2232 | 0.195 |
| models | BRFSS | reg. logistic, lambda=1e-06 | 124 | 0.4231 ± 0.0042 | 0.8696 | 0.2232 | 0.195 |
| models | BRFSS | reg. logistic, lambda=1e-05 | 124 | 0.4233 ± 0.0040 | 0.8696 | 0.2229 | 0.195 |
| models | BRFSS | reg. logistic, lambda=0.0001 | 124 | 0.4234 ± 0.0043 | 0.8697 | 0.2214 | 0.196 |
| models | BRFSS | reg. logistic, lambda=0.001 | 124 | 0.4235 ± 0.0032 | 0.8719 | 0.2134 | 0.206 |
| ablation | simple (mean imputation) | reg. logistic, lambda=0.001 | 315 | 0.4126 ± 0.0036 | 0.8661 | 0.1636 | 0.207 |
| ablation | BRFSS | reg. logistic, lambda=0.001, undersampled | 124 | 0.4224 ± 0.0038 | 0.8708 | 0.3709 | 0.728 |
| ablation | BRFSS + target-corr selection | reg. logistic, lambda=0.001 | 112 | 0.4238 ± 0.0039 | 0.8701 | 0.2124 | 0.202 |
| ablation | BRFSS + squares | reg. logistic, lambda=0.001 | 157 | 0.4232 ± 0.0031 | 0.8716 | 0.2133 | 0.206 |
| ablation | BRFSS + one-hot | reg. logistic, lambda=0.001 | 350 | 0.4261 ± 0.0035 | 0.8702 | 0.2216 | 0.204 |
| ablation | BRFSS + one-hot + squares | reg. logistic, lambda=0.001 | 383 | 0.4262 ± 0.0034 | 0.8706 | 0.2228 | 0.205 |
