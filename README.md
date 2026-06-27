# Deep Tangency Portfolio

This repository contains the replication code for Feng, Jiang, Li, Song, and Wang,  
**Deep Tangency Portfolio**.

The main replication directory is `MS_replication/`. All paths below are relative to that directory unless otherwise noted.

---

## How to Replicate the Results

### 1. For Linux Users
- In the command line, navigate to the directory: `cd ./MS_replication/`.
- Make sure the full processed input files and precomputed model-output folders are placed in the expected paths under `./data/` and `./code/results_*/`.
- Run the script: `sh submit_whole.sh`.
- The script runs `./code/final_results_with_batch.py` and writes the generated tables and figures to `./output/`.
- The authors ran the final result-generation code using Python 3.6.13, PyTorch 1.10.2, and parallel computation on a server with 96 Intel(R) Xeon(R) Gold 6230 @ 2.10GHz CPUs and 314 GB of RAM. The final table/figure generation takes approximately **1.8 minutes**.

### 2. For Windows/Mac Users
- Set the working directory to `./MS_replication/code/`.
- Run `python3 final_results_with_batch.py`.
- The output files will be written to `./MS_replication/output/`.

### 3. Generated Tables and Figures
- **Table 1**: `Table_1_panelA.csv`, `Table_1_panelB.csv`, and `Table_1_panelC.csv`
- **Table 2 Panel B**: `Table_2_panelB.csv`
- **Table 3**: `Table_3.csv`
- **Table 4**: `Table_4.csv`
- **Table 5**: `Table_5.csv`
- **Figure 2**: `Figure_2.pdf`
- **Figure 3**: `Figure_3.pdf`
- **Figure 4**: `Figure_4.pdf`

### 4. Software
The authors also tested the final result-generation scripts using Python 3.11.3 with the following package versions:

- `pandas` 1.5.3
- `numpy` 1.24.3
- `scipy` 1.10.1
- `statsmodels` 0.13.5
- `matplotlib` 3.11.0
- `pyarrow` 11.0.0
- `seaborn` 0.12.2
- `torch` 2.0.1

---

## Data Availability and Provenance

This paper uses publicly available data sources, WRDS data, and author-processed data files:

- Corporate bond transaction and issuance raw data are downloaded from WRDS:
  - Bond transactions data from TRACE Enhanced provided by FINRA:  
    [https://wrds-www.wharton.upenn.edu/pages/get-data/otc-corporate-bond-and-agency-debt-bond-transaction-data/](https://wrds-www.wharton.upenn.edu/pages/get-data/otc-corporate-bond-and-agency-debt-bond-transaction-data/)
  - Bond Issuers, Bond Issues, and Bond Ratings data from LSEG Mergent Fixed Income Securities Database (FISD):  
    [https://wrds-www.wharton.upenn.edu/pages/get-data/lseg-mergent/](https://wrds-www.wharton.upenn.edu/pages/get-data/lseg-mergent/)
- Bond-equity linking information is obtained from the WRDS Bond CRSP Link:  
  [https://wrds-www.wharton.upenn.edu/pages/get-data/linking-suite-wrds/bond-crsp-link/](https://wrds-www.wharton.upenn.edu/pages/get-data/linking-suite-wrds/bond-crsp-link/)
- Bond term and default factor inputs use Amit Goyal's website and the updated `PredictorMacroMonthly.csv` file:  
  [https://sites.google.com/view/agoyal145/home](https://sites.google.com/view/agoyal145/home)
- Treasury bond data are from the Federal Reserve FEDS Treasury yield curve data:  
  [https://www.federalreserve.gov/econres/feds/the-us-treasury-yield-curve-1961-to-the-present.htm](https://www.federalreserve.gov/econres/feds/the-us-treasury-yield-curve-1961-to-the-present.htm)
- The CBOE VIX series is obtained from Yahoo Finance through the R `quantmod` package:  
  [https://finance.yahoo.com/quote/%5EVIX?p=%5EVIX](https://finance.yahoo.com/quote/%5EVIX?p=%5EVIX)
- Macroeconomic uncertainty data are from Sydney C. Ludvigson's website:  
  [https://www.sydneyludvigson.com/macro-and-financial-uncertainty-indexes](https://www.sydneyludvigson.com/macro-and-financial-uncertainty-indexes)

Equity and option characteristics are included in the processed merged characteristic files. The package uses the final merged data directly rather than re-creating these raw data from the original vendor files. Final processed inputs are stored in:

- `./MS_replication/data/data_in_all_final/`
- `./MS_replication/data/ipca_final/`
- `./MS_replication/data/rppca_final/`

If using this GitHub repository without the complete replication-package files, please place the large processed characteristic panels and precomputed training outputs in the corresponding paths before running the scripts. In particular, `final_results_with_batch.py` expects the processed `.feather` characteristic files under `./MS_replication/data/data_in_all_final/` and the precomputed `results_*` folders under `./MS_replication/code/`.

---

## Reference

- Welcome to cite our paper.

```bibtex
@unpublished{feng2026deep,
  title  = {Deep Tangency Portfolio},
  author = {Feng, Guanhao and Jiang, Liang and Li, Junye and Song, Yizhi and Wang, Yuanzhi},
  note   = {Working paper},
  year   = {2026}
}
```

## Contact

For questions or comments regarding the replication code, please contact:

- **Yuanzhi Wang**
- Email: yuanzwang5-c@my.cityu.edu.hk
