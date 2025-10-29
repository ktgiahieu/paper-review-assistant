This experiment is to compare the

- original Llama-3.1-70B \cite{grattafiori2024llama}
- fine-tuned CycleReviewer-Llama-3.1-70B \cite{weng2025cycleresearcher} aimed to review on ICLR

We will use:

- Data: ICLR 2024 pairs of (submitted, camera-ready) papers. We get it by checking the ICLR2024.csv `arxiv_info` columns to see if the date is before Jan 15 2024 as submitted, and > Jan 15 2024 as camera-ready version. ( we don't have any data point that is Jan 15 2024 in our dataset).
- Method: vllm to host the 2 models above (3 runs each) to:
  - Generate textual reviews and numerical scores
  - Compare:
    - Manually the textual review between submitted and camera-ready version
    - Check the MAE between the pairs of mean score and std for the pairs of (3 reviews for submitted, 3 reviews for camera-ready)
