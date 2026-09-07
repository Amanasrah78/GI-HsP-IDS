# MQTTset Classical Baseline Results

Dataset protocol:
- MQTTset native reduced split
- Train: 231,646 samples
- Test: 99,290 samples
- 20 processed features
- mqtt.msg excluded
- zero/near-zero variance features removed
- categorical mappings fitted on training data
- MinMaxScaler fitted on training data

## Results

| Model | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | MCC |
|---|---:|---:|---:|---:|---:|
| RF-balanced | 0.6530 | 0.6561 | 0.5817 | 0.7794 | 0.5892 |
| RF-standard | 0.8419 | 0.6459 | 0.7255 | 0.8361 | 0.7445 |
| HistGradientBoosting | 0.9233 | 0.7285 | 0.7837 | 0.9211 | 0.8697 |

Notes:
- RF-balanced heavily overpredicts the rare flood class.
- RF-standard provides a more realistic baseline.
- HGB is the strongest classical baseline under this protocol.
- These values are not directly comparable to MFTST published results because the preprocessing/split/windowing protocols differ.
