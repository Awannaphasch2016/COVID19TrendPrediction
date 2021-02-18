#/bin/bash

cd /home/awannaphasch2016/Documents/Working/COVID19TrendPrediction/

if [ ! -f Data/Raw/COVID19Cases/StateLevels/us-states.csv ] 
then
    echo "installing from https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv ..."
    mkdir -p Data/Raw/COVID19Cases/StateLevels/
    wget https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv -O  Data/Raw/COVID19Cases/StateLevels/us-states.csv 
else
    echo "File already exist at Data/Raw/COVID19Cases/StateLevels/us-states.csv"
fi

