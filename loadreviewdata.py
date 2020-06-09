from dependencies import *

review_data = pd.read_csv('../processeddata/review22feb.csv', low_memory=False)
review_data['date'] = pd.to_datetime(review_data['date'])

