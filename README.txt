An Interactive Choropleth to the Visualize Sentiment Analysis of COVID-19 Tweets Worldwide

DESCRIPTION 
This codebase displays a worldwide choropleth of the a sentiment analysis and classification of fake tweets over the course of the COVID-19 pandemic. Using this visualization, the user will be able to see how sentiments are averaged over any week time period from 2/12/2020 to 9/12/2022. The user will also be able to select what feature they would like to view from the dataset. There are four available features for the user: sentiment score, weighted sentiment score by the number of retweets, the number of fake tweets, and the confidence of the fake tweet classification. 

There are three sections of our work: getting the covid related tweets, analysis on tweets, and visualization. To get the covid tweets, the get_hydratory.py script has to be run and after running, the results will be stored in the data folder. To get the tweet text, the twitter hydrator should be used in conjunction with the hydrator_input.csv file in the data directory. The twitter hydrator installation can be found here: https://github.com/DocNow/hydrator . With the pulled data from the hydrator in the pulled data folder, run the get_tweet.py script. The resulting tweets and their necessary information will be stored in the data directory.

To run the analysis, the run_roberta.py and the run_vader.py script have to be run. The resulting analysis will be stored in the data folder. To aggregate the results, run the score_cleaning.py script on the resulting data. 

To see the visualization, none of the previous scripts are required. The results from all of the previous methods are aggregated into a document called final_set.csv and clean_agg_set.csv. To see the choropleth, the user just has to run the choropleth_visualization.py script. 


INSTALLATION 

Utilizing the Anaconda package manager [https://docs.anaconda.com/], the environment can be installed. 

```bash
conda env create -f environment.yml
```

EXECUTION 

To run the visualization, use the following code in the SRC directory. 

```bash
python choropleth_visualization.py
```

