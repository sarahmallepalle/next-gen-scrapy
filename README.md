# Intro to `next-gen-scrapy`

## Summary

This is the first version released of `next-gen-scrapy`. This repo of was built to allow users to scrape all of the regular season and postseason pass charts from NextGenStats from 2017 onwards, and extract all passes -  completions, incompletions, interceptions, and touchdowns - from the pass charts. The file `pass_and_game_data.csv` contains the final version of all of the data after all of the Python and R scripts are run, including Game ID, home team, away team, week, season, player, type of pass, and pass location for the 2017 season, and up to week 16 of the 2018 season. This repo will be maintained regularly for bug fixes and new, exciting features and updates! Thank you to Sam Ventura, Kostas Pelechrinis, and Ron Yurko for all your help and guidance with this project!

### Example Cleaned Pass Chart (with axes in yards) - Nick Foles in Super Bowl LII

![Nick Foles in Super Bowl LII](https://raw.githubusercontent.com/sarahmallepalle/next-gen-scrapy/master/axes.jpg)

## Data

Column | Definition
---|---------
`game_id` | NFL GameID
`team` | Pass chart's team
`name` | Pass chart's player first and last name 
`pass_type` | COMPLETE, INCOMPLETE, INTERCEPTION, or TOUCHDOWN
`x` | x-coordinate of field location in yards; -26.66 <= x <= 26.66, with x = 0 as the vertical axis in the center of the field. 
`y` | y-coordinate of field location in yards; -10 <= y <= 75, with y = 0 as the horizontal axis at the Line of Scrimmage
`type` | Regular ("reg") or postseason ("post") game
`home_team` | Home team of game
`away_team` | Away team of game
`week` | Week of game
`season` | Year of game - 2017 or 2018 (...for now :) )

## Installation

Requires Python 2.7 and R

Python 2.7:
```
pip install --upgrade pip
pip install requests bs4 lxml Pillow opencv-python scipy numpy pandas scikit-learn`
```

R:
```
install.packages(c("devtools", "dplyr"))
devtools::install_github(repo = "ryurko/nflscrapR")
```

## Usage

From the command line:

```
python scrape.py 	# Scrape images and html data, store in folder Pass_Charts
```
```
python clean.py 	# Clean images and html data, store in folder Cleaned_Pass_Charts
```
```
python main.py 		# Extract pass information from images and data, output to pass_locations.csv
```
```
Rscript game_data_from_nflscrapR.R 	# Use nflscrapR to match pass information to game information, output to pass_and_game_data.csv
```

## Known bugs as of this first version
- For a significant amount of pass charts on Next Gen Stats, the number of incomplete passes given in the HTML data, does not match the actual number of incomplete passes depicted in the Pass Charts (possibly because these passes are thrown out of bounds?) DBSCAN was used to try to make up for the mismatch in data, but may be slightly off in calculating number of incomplete passes present in the image. Example image here:  https://nextgenstats.nfl.com/charts/list/all/kansas-city-chiefs/2017/wild-card/alex-smith/SMI031126/2017/wild-card/pass (there are supposed to be 33-24=9 incompletes in the pass chart, but there are only 8 shown on the field.)
- NA values if pass locations could not be extracted. 
- A row of NA values next to a Game ID if a pass chart could not be extracted from the html.
