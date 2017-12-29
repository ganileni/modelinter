modelinter
==============================

Study about model interconnectedness. This repository contains the code used to execute the calculations presented in the paper **"Why Model Interconnectedness Matters -  and a Technique to Approach it"** *[(SSRN:3078021)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3078021 "Link to the paper")*.

Notes on the code:

The bulk of the code is organized in a module called `modelinter`, which is extensively commented.
Most of the logic is in the files contained under the `modelinter/models` folder, data is found in `modelinter/resources`. The code has been tested for both Python 3.5 and 3.6.

The code is explained in 4 notebooks contained in the `notebooks/` folder. Each of them starts with a number.
They are meant to be read in increasing order, to help understanding what the code does and how it's structured.
We advice to start from them, and then move to the rest of the code if you want a more in-depth understanding.
The notebooks will work straight away, because we saved the necessary subset of the data in the data/raw folder.#
Notice, however, that the calculations might take quite a lot. If you don't want to be stuck waiting, you might want to execute overnight the script `models/run_all_calculations.py`.
This will execute all the calculations and pickle the results so that, when you run the notebooks, they will directly read from the pickles and print the results, with no waiting time.

If you want the complete dataset, we suggest running `/notebooks/download-dataset.ipynb`, but it's the part of the code we tested the least, as it is unnecessary to reproduce the resuts. A lower level representation of the repository is found below, in the "Project Organization" section.



Project Organization
------------
```
/
├── LICENSE
├── modelinter
│   ├── models
│   │   ├── calculations.py <- all the calculation logic is here
│   │   ├── constants.py <- constants for the model and other immutable stuff
│   │   ├── model_results.py <- code for plots in notebook 2
│   │   ├── model_tuning.py <- code for plots in notebook 3
│   │   ├── pgm.py <- objects that encode the models
│   │   ├── run_all_calculations.py <- script that runs all the calculations and pickles the results
│   │   └── utils.py <- miscellaneous utilites
│   ├── notebooks <- each notebook has an associated script in this folder, in case you dont' want to use jupyter
│   │   ├── 0-preliminary_analysis.ipynb <- analysis on the properties of the dataset
│   │   ├── 1-model_setup.ipynb <- how the model (and the code) works
│   │   ├── 2-model_results.ipynb <- the results of the model as shown in the paper
│   │   ├── 3-model_tuning.ipynb <- explores sensitivity of the model to parameters change
│   │   ├── download-dataset.ipynb <- downloads the data from alpha vantage
│   │   └── imports.py <- a script run by each notebook that does some basic imports
│   ├── preprocessing
│   │   └── imports_load_data.py <- all disk reading operations used in the code
│   ├── resources
│   │   ├── data
│   │   │   ├── interim <- this will store any pickled results
│   │   │   ├── processed
│   │   │   │   └── free_subset
│   │   │   │       ├── timeseries.csv
│   │   │   │       └── timeseries_returns.csv
│   │   │   └── raw <- immutable data dump
│   │   │       ├── alphavantage_apikey.txt <- if you want to run download-dataset.ipynb you need to get an alpha vantage API key and put it in this file
│   │   │       ├── FedSeverelyAdverse.csv <- csv with CCAR scenario
│   │   │       └── sp500_constituents.csv <- constituents of SP500 at the time when the dataset was downloaded
│   │   └── plots <- the plots published in the paper
│   └── test <- tests for the code
│       ├── test_notebook1.py
│       └── test_regressions.py
├── README.md
├── references
│   └── CCAR _2017.pdf <- CCAR document released by the FED
└── requirements.txt
```


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

<p><small>we acknowledge AlphaVantage for the data used in this study. It is freely downloadable at https://www.alphavantage.co/</small></p>

<p><small>This code is distributed under the MIT license. See [/LICENSE][LICENSE]</small></p>
