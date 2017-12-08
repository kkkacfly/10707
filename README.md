This repository contains the codes about the implementation of our CNN-rLSTM models on the TRANCOS dataset for 10707 project. The architecture of the code:
    - Data preprocessing:
        - preprocessing_function.py
    - Loss and operation functions:
        - util.py
    - Model initialization and training:
        - CNN_baseline.py: CNN only, with density output
        - CNN_baseline_residual.py: CNN with residual connection, with density and count output
        - CNN_rLSTM.py: time distributed CNN with LSTM with residual connection, with density and count output
