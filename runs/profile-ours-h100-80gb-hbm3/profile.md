## Profile (NVIDIA H100 80GB HBM3, backbone = `language_model.model.language_model`, 7.94B of 9.53B params, gdn_fast_path=False)

| record | tokens | encode | collate | pre | backbone | head | post | total ms | backbone share |
|---|---|---|---|---|---|---|---|---|---|
| card_trio | 300 | 1.8 | 0.2 | 0.1 | 128.6 | 4.6 | 0.1 | 135.4 | 95% |
| pulse-11714037 | 843 | 2.3 | 0.3 | 0.1 | 219.2 | 4.5 | 0.1 | 226.7 | 97% |
| long_2k | 2486 | 5.2 | 0.6 | 0.1 | 353.0 | 5.2 | 0.1 | 364.4 | 97% |
