This the code and model for our paper: Clarity Contrast and Similarity Selection for \\ Multi-Focus Image Fusion

Make sure the required environment is properly set up before running.
Please run the following command in the terminal to test our CSNet:

CUDA_VISIBLE_DEVICES=0 python test.py

The terminal will output the average evaluation metrics on the Lytro, MFFW, MFI-WHU and SIMIF datasets. 
The fused images will be saved in the "result" directory.

Note that the original implementations of the six evaluation metrics provided in the paper are all in MATLAB. 
We have reproduced four of these metrics in Python, including Q_MI, Q_NCIE, Q_G and Q_Y.
The remaining two metrics Q_C and Q_CB are designed using toolboxes in MATLAB and can only be computed in MATLAB. 
If you need to test the corresponding metrics, please save the fusion results and compute them in MATLAB.