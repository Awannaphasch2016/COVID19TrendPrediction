#!/bin/bash
# export PYTHONPATH="/home/awannaphasch2016/Documents/Working/CovidTrendPrediction/:$PYTHONPATH"
# export PYTHONPATH="/home/awannaphasch2016/Documents/Working/Covid19TrendPrediction/:$PYTHONPATH"
export PYTHONPATH="$HOME/Documents/Working/COVID19TrendPrediction/:$PYTHONPATH"
export PYTHONPATH="$HOME/Documents/Working/Covid19TrendPrediction/:$PYTHONPATH"
# source .venv/bin/activate
# pip3 install -r requirements.txt

# download data
cd $HOME/Documents/Working/COVID19TrendPrediction/
export WANDB_API_KEY=142654f03237fa46e04cc0372da5c045c64574de 
export AWS_ACCESS_KEY_ID="AKIA3ZMDQYM6TSTIV6GN"
export AWS_SECRET_ACCESS_KEY="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"



./Scripts/Setups/download_data.sh
 
