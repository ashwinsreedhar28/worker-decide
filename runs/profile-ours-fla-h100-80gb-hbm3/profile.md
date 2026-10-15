## Profile (NVIDIA H100 80GB HBM3, backbone = `language_model.model.language_model`, 7.94B of 9.53B params, gdn_kernels={'fla': True, 'causal_conv1d': False})

| record | tokens | encode | collate | pre | backbone | head | post | total ms | backbone share |
|---|---|---|---|---|---|---|---|---|---|
| card_trio | 300 | 1.7 | 0.2 | 0.1 | 43.6 | 4.5 | 0.1 | 50.3 | 87% |
| pulse-11714037 | 843 | 2.0 | 0.3 | 0.1 | 50.3 | 4.1 | 0.1 | 57.0 | 88% |
| long_2k | 2486 | 4.9 | 0.6 | 0.1 | 90.1 | 4.9 | 0.1 | 100.8 | 89% |
