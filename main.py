import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import tensorflow as tf
from colorama import Fore, Style, just_fix_windows_console

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.DataProviders.UpToDateDataManager import UpToDateDataManager
from src.Features.feature_engineering import augment_features
from src.Predict import NN_Runner, XGBoost_Runner
from src.Training.AutoTrainer import retrain_models
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games

TODAYS_GAMES_URL = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{year}/scores/00_todays_scores.json'
DATA_URL_TEMPLATE = ('https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&Height=&ISTRound='
    '&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame'
    '&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={season}&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange='
    '&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=')

SCHEDULE_FILE = Path(__file__).resolve().parent / 'Data' / 'nba-2025-UTC.csv'
DATE_FORMAT = '%d/%m/%Y %H:%M'
DEFAULT_DAYS_OFF = 7
SPORTSBOOK_CHOICES = (
    'fanduel',
    'draftkings',
    'betmgm',
    'pointsbet',
    'caesars',
    'wynn',
    'bet_rivers_ny'
)


def _current_season_code(reference: datetime | None = None) -> str:
    reference = reference or datetime.today()
    start_year = reference.year if reference.month >= 9 else reference.year - 1
    end_year = str((start_year + 1) % 100).zfill(2)
    return f"{start_year}-{end_year}"


def _todays_games_feed(reference: datetime | None = None) -> str:
    reference = reference or datetime.today()
    return TODAYS_GAMES_URL.format(year=reference.year)


def _load_schedule(schedule_path: Path = SCHEDULE_FILE) -> pd.DataFrame:
    """Return the master schedule with parsed dates."""
    if not schedule_path.exists():
        raise FileNotFoundError(
            f"NBA schedule file not found at {schedule_path}. Please run the data preparation step."
        )
    schedule_df = pd.read_csv(schedule_path)
    schedule_df['Date'] = pd.to_datetime(schedule_df['Date'], format=DATE_FORMAT)
    return schedule_df


def _days_since_last_game(team: str, schedule_df: pd.DataFrame, reference_time: datetime) -> int:
    """Calculate the number of days since the provided team last played."""
    team_games = schedule_df[(schedule_df['Home Team'] == team) | (schedule_df['Away Team'] == team)]
    recent_games = team_games.loc[team_games['Date'] <= reference_time].sort_values('Date', ascending=False)
    if recent_games.empty:
        return DEFAULT_DAYS_OFF
    last_game_date = recent_games.iloc[0]['Date']
    return (timedelta(days=1) + reference_time - last_game_date).days


def createTodaysGames(games, df, odds, schedule_df):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []

    home_team_days_rest = []
    away_team_days_rest = []

    reference_time = datetime.today()
    for game in games:
        home_team = game[0]
        away_team = game[1]
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        if odds is not None:
            game_odds = odds[home_team + ':' + away_team]
            todays_games_uo.append(game_odds['under_over_odds'])

            home_team_odds.append(game_odds[home_team]['money_line_odds'])
            away_team_odds.append(game_odds[away_team]['money_line_odds'])

        else:
            todays_games_uo.append(input(home_team + ' vs ' + away_team + ': '))

            home_team_odds.append(input(home_team + ' odds: '))
            away_team_odds.append(input(away_team + ' odds: '))

        # calculate days rest for both teams
        home_days_off = _days_since_last_game(home_team, schedule_df, reference_time)
        away_days_off = _days_since_last_game(away_team, schedule_df, reference_time)

        home_team_days_rest.append(home_days_off)
        away_team_days_rest.append(away_days_off)
        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)].rename(
            lambda col: f"{col}.1" if not str(col).endswith('.1') else str(col)
        )
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_days_off
        stats['Days-Rest-Away'] = away_days_off
        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T
    games_data_frame = augment_features(games_data_frame)

    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'], errors='ignore')
    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def _prompt_yes_no(message: str, default: bool = False) -> bool:
    prompt_hint = 'Y/n' if default else 'y/N'
    while True:
        response = input(f"{message} [{prompt_hint}]: ").strip().lower()
        if not response:
            return default
        if response in ('y', 'yes'):
            return True
        if response in ('n', 'no'):
            return False
        print('Please respond with yes or no.')


def _prompt_for_model_selection():
    print('\nSelect which model(s) to run:')
    print('  1) XGBoost (faster, tree-based)')
    print('  2) Neural Network (normalized inputs)')
    print('  3) Run both models')
    while True:
        choice = input('Enter choice [1-3]: ').strip()
        if choice in {'1', '2', '3'}:
            return choice
        print('Please pick 1, 2 or 3.')


def _prompt_for_sportsbook():
    choices = ', '.join(SPORTSBOOK_CHOICES)
    sportsbook = input(
        f"Enter sportsbook to automatically fetch odds ({choices}) or press Enter to enter odds manually: "
    ).strip().lower()
    return sportsbook if sportsbook in SPORTSBOOK_CHOICES else ''


def _ensure_interactive_options(args):
    """Fill in missing CLI options by prompting the user when possible."""
    if not sys.stdin.isatty():
        return args

    missing_models = not any((args.xgb, args.nn, args.A))
    missing_odds = args.odds is None
    missing_kc = not args.kc

    if not any((missing_models, missing_odds, missing_kc)):
        return args

    print('\nInteractive mode detected - please answer the prompts below.')
    if missing_models:
        selection = _prompt_for_model_selection()
        if selection == '1':
            args.xgb = True
        elif selection == '2':
            args.nn = True
        else:
            args.A = True

    if missing_odds:
        sportsbook = _prompt_for_sportsbook()
        args.odds = sportsbook if sportsbook else None

    if missing_kc:
        args.kc = _prompt_yes_no('Show Kelly Criterion staking suggestions?', default=False)

    return args


def _maybe_refresh_data(args):
    if not args.refresh_data:
        return
    sportsbook = args.odds if args.odds else 'fanduel'
    print(f"Refreshing dataset with the latest {args.refresh_window} day(s) of results using {sportsbook} odds...")
    manager = UpToDateDataManager(season=args.season)
    new_rows = manager.refresh_recent_data(days_back=args.refresh_window, sportsbook=sportsbook)
    if new_rows.empty:
        print('No new completed games found during refresh.')
        return
    if args.skip_retrain:
        print('Data refreshed. Skipping retraining per flag.')
        return
    print('Retraining models so predictions reflect the new information...')
    paths = retrain_models()
    XGBoost_Runner.refresh_models(paths.get('xgb_ml'), paths.get('xgb_ou'))
    NN_Runner.refresh_models(paths.get('nn_ml'), paths.get('nn_ou'))


def main(args):
    _maybe_refresh_data(args)
    todays_feed = _todays_games_feed()
    odds = None
    if args.odds:
        odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
        games = create_todays_games_from_odds(odds)
        if len(games) == 0:
            print("No games found.")
            return
        if (games[0][0] + ':' + games[0][1]) not in list(odds.keys()):
            print(games[0][0] + ':' + games[0][1])
            print(Fore.RED,"--------------Games list not up to date for todays games!!! Scraping disabled until list is updated.--------------")
            print(Style.RESET_ALL)
            odds = None
        else:
            print(f"------------------{args.odds} odds data------------------")
            for g in odds.keys():
                home_team, away_team = g.split(":")
                print(f"{away_team} ({odds[g][away_team]['money_line_odds']}) @ {home_team} ({odds[g][home_team]['money_line_odds']})")
    else:
        data = get_todays_games_json(todays_feed)
        games = create_todays_games(data)
    data_url = DATA_URL_TEMPLATE.format(season=args.season)
    data = get_json_data(data_url)
    df = to_data_frame(data)
    try:
        schedule_df = _load_schedule()
    except FileNotFoundError as exc:
        print(Fore.RED + str(exc) + Style.RESET_ALL)
        return
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = createTodaysGames(games, df, odds, schedule_df)
    if args.nn:
        print("------------Neural Network Model Predictions-----------")
        data = tf.keras.utils.normalize(data, axis=1)
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
    if args.xgb:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
    if args.A:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
        data = tf.keras.utils.normalize(data, axis=1)
        print("------------Neural Network Model Predictions-----------")
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")


if __name__ == "__main__":
    just_fix_windows_console()
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from. (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    parser.add_argument('--refresh-data', action='store_true', help='Download the latest games before running predictions')
    parser.add_argument('--refresh-window', type=int, default=3, help='How many recent days to pull when refreshing data')
    parser.add_argument('--skip-retrain', action='store_true', help='Skip retraining even if new data was downloaded')
    parser.add_argument('--season', default=_current_season_code(), help='Season identifier used for NBA Stats queries (e.g. 2025-26)')
    parsed_args = parser.parse_args()
    parsed_args = _ensure_interactive_options(parsed_args)
    if parsed_args.odds:
        parsed_args.odds = parsed_args.odds.lower()
    main(parsed_args)
