from dependencies import *

user_data = pd.read_csv('../data/csv/user.csv', low_memory=False)
user_data = user_data.fillna('NaN')
elite_udata  = user_data.loc[user_data['elite'] != 'NaN']
non_elite_udata  = user_data.loc[user_data['elite'] == 'NaN']
print('User Data Loaded')

business_data = pd.read_csv('../data/csv/business.csv', low_memory=False)
rest_idx = ['Restaurants' in x.split(', ') for x in list(business_data['categories'].replace(np.nan, '', regex=True))]
rest_data = business_data[rest_idx]
print('Businesses Data Loaded')

from loadreviewdata import review_data
print('Review Data Loaded')
