# NBA Sports Betting Using Machine Learning 🏀
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/output.png" width="1010" height="292" />

A machine learning AI used to predict the winners and under/overs of NBA games. Takes all team data from the 2007-08 season to current season, matched with odds of those games, using a neural network to predict winning bets for today's games. Achieves ~69% accuracy on money lines and ~55% on under/overs. Outputs expected value for teams money lines to provide better insight. The fraction of your bankroll to bet based on the Kelly Criterion is also outputted. Note that a popular, less risky approach is to bet 50% of the stake recommended by the Kelly Criterion.
## Packages Used

Use Python 3.11. In particular the packages/libraries used are...

* Tensorflow - Machine learning library
* XGBoost - Gradient boosting framework
* Numpy - Package for scientific computing in Python
* Pandas - Data manipulation and analysis
* Colorama - Color text output
* Tqdm - Progress bars
* Requests - Http library
* Scikit_learn - Machine learning library

## Usage

<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Expected_value.png" width="1010" height="424" />

Make sure all packages above are installed.

```bash
$ git clone https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting.git
$ cd NBA-Machine-Learning-Sports-Betting
$ pip3 install -r requirements.txt
$ python3 main.py -xgb -odds=fanduel
```

Odds data will be automatically fetched from sbrodds if the -odds option is provided with a sportsbook.  Options include: fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny

If `-odds` is not given, enter the under/over and odds for today's games manually after starting the script.

Optionally, you can add '-kc' as a command line argument to see the recommended fraction of your bankroll to wager based on the model's edge.

### Interactive / Windows usage

Running `python main.py` without any flags will now launch an interactive prompt that works the same on macOS/Linux terminals and the Windows Command Prompt/PowerShell.  The prompt helps you:

* Choose whether to run the XGBoost model, the neural network, or both.
* Pick a sportsbook (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny) for automatically scraped odds or fall back to manual entry.
* Decide if you want to display Kelly Criterion staking recommendations.

You can still pass the original CLI flags when scripting or running in environments without an interactive terminal.

### Keeping the data fresh

Daily lines shift quickly, so `main.py` now has a turnkey workflow for downloading the latest box scores/odds, updating the
training dataset, and retraining both the neural network and XGBoost models before producing predictions:

```bash
# pull the last three days of results, refresh the SQLite dataset, retrain models, then run both predictors
python main.py -A --refresh-data --refresh-window 3 -odds=fanduel
```

Key switches:

* `--refresh-data` – grab the most recent NBA results (using the selected sportsbook for totals/moneylines), append them to the
  historical dataset, and inject new composite features such as efficiency and rebound/pace differentials.
* `--refresh-window <days>` – control how far back the refresh looks (default: 3 days).
* `--skip-retrain` – if you only want to update the dataset cache (for example before running the standalone training scripts)
  you can opt out of on-the-fly retraining.
* `--season` – override the NBA season string passed to stats.nba.com (defaults to the current season, e.g. `2025-26`).

Behind the scenes the refresh pipeline stores the merged dataset in `Data/dataset.sqlite` and then writes the most recent models
to `Models/XGBoost_Models/latest_*.json` and `Models/NN_Models/latest_*.keras`. If these files exist they are loaded
automatically, otherwise the repo falls back to the previously committed weights.

## Flask Web App
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Flask-App.png" width="922" height="580" />

This repo also includes a small Flask application to help view the data from this tool in the browser.  To run it:
```
cd Flask
flask --debug run
```

## Getting new data and training models
```
# Create dataset with the latest data for 2023-24 season
cd src/Process-Data
python -m Get_Data
python -m Get_Odds_Data
python -m Create_Games

# Train models
cd ../Train-Models
python -m XGBoost_Model_ML
python -m XGBoost_Model_UO
```

## Contributing

All contributions welcomed and encouraged.
