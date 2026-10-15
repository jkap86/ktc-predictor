"""v5 what-if predictor: season points in, end-of-season dynasty SF value out.

Inputs are a player's start-of-season value, age, position and the season
fantasy points you expect them to score. The output is the predicted
end-of-season value on the same scale as the label source it was trained on.

Labels are swappable (see ``labels.py``): KTC today, Backchannel
``reference_value`` once the past-season backfill lands.
"""
