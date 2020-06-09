from dependencies import *

def create_dataset_distribution(min_thold, bdata, bid_2_stars, bid_2_stars_e, max_thold=float('inf')):
    basicprint = False
    x1 = []
    x2 = []
    y = []
    bids = bdata['business_id'].values
    for bid in bids:
        if bid in bid_2_stars and bid in bid_2_stars_e:
            stars_e = bid_2_stars_e[bid]
            stars_ne = bid_2_stars[bid]
            lestars = len(stars_e)
            if(lestars >= min_thold and lestars < max_thold):
                x = []
                countratings = [Counter(stars_ne)[i] for i in range(1,6)]
                distribution = [x/sum(countratings) for x in countratings]
                x.append(distribution)
                x = [item for sublist in x for item in sublist]
                x1.append(x)
                y.append(np.mean(stars_e))                        
    xfeat = np.array(x1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    xfeat = imp.fit_transform(xfeat)
    yfeat = np.array(y).reshape(-1,)
    print("Dimension of Our Dataset: ", xfeat.shape)
    return xfeat,yfeat
  

def create_dataset_distribution_and_avg(min_thold, bdata, bid_2_stars, bid_2_stars_e, max_thold=float('inf')):
    basicprint = False
    x1 = []
    x2 = []
    y = []
    bids = bdata['business_id'].values
    for bid in bids:
        if bid in bid_2_stars and bid in bid_2_stars_e:
            stars_e = bid_2_stars_e[bid]
            stars_ne = bid_2_stars[bid]
            lestars = len(stars_e)
            if(lestars >= min_thold and lestars < max_thold):
                x = []
                countratings = [Counter(stars_ne)[i] for i in range(1,6)]
                distribution = [x/sum(countratings) for x in countratings]
                x.append(distribution)
                x.append([np.mean(stars_ne)])
                x = [item for sublist in x for item in sublist]
                x1.append(x)
                y.append(np.mean(stars_e))                        
    xfeat = np.array(x1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    xfeat = imp.fit_transform(xfeat)
    yfeat = np.array(y).reshape(-1,)
    print("Dimension of Our Dataset: ", xfeat.shape)
    return xfeat,yfeat

  
def give_categ(avg):
    temp = np.zeros(41)
    c = int((avg-1)/0.1) 
    temp[c] = 1
    return temp
  
  
def create_dataset_buckets(min_thold, bdata, bid_2_stars, bid_2_stars_e, max_thold=float('inf')):
    basicprint = False
    x1 = []
    x2 = []
    y = []
    bids = bdata['business_id'].values
    for bid in bids:
        if bid in bid_2_stars and bid in bid_2_stars_e:
            stars_e = bid_2_stars_e[bid]
            stars_ne = bid_2_stars[bid]
            lestars = len(stars_e)
            if(lestars >= min_thold and lestars < max_thold):
                x = []
                x.append(give_categ(np.mean(stars_ne)))
                x = [item for sublist in x for item in sublist]
                x1.append(x)
                y.append(np.mean(stars_e))                        
    xfeat = np.array(x1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    xfeat = imp.fit_transform(xfeat)
    yfeat = np.array(y).reshape(-1,)
    print("Dimension of Our Dataset: ", xfeat.shape)
    return xfeat,yfeat
  


def create_dataset_buckets_and_distri(min_thold, bdata, bid_2_stars, bid_2_stars_e, max_thold=float('inf')):
    basicprint = False
    x1 = []
    x2 = []
    y = []
    bids = bdata['business_id'].values
    for bid in bids:
        if bid in bid_2_stars and bid in bid_2_stars_e:
            stars_e = bid_2_stars_e[bid]
            stars_ne = bid_2_stars[bid]
            lestars = len(stars_e)
            if(lestars >= min_thold and lestars < max_thold):
                x = []
                x.append(give_categ(np.mean(stars_ne)))
                countratings = [Counter(stars_ne)[i] for i in range(1,6)]
                distribution = [x/sum(countratings) for x in countratings]
                x.append(distribution)
                x = [item for sublist in x for item in sublist]
                x1.append(x)
                y.append(np.mean(stars_e))                        
    xfeat = np.array(x1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    xfeat = imp.fit_transform(xfeat)
    yfeat = np.array(y).reshape(-1,)
    print("Dimension of Our Dataset: ", xfeat.shape)
    return xfeat,yfeat
  

def create_dataset_everything(min_thold, bdata, bid_2_stars, bid_2_stars_e, bid_2_textenc, max_thold=float('inf')):    
    basicprint = False
    x1 = []
    x2 = []
    y = []
    bids = bdata['business_id'].values
    for bid in bids:
        if bid in bid_2_stars and bid in bid_2_stars_e and bid in bid_2_textenc:
            stars_e = bid_2_stars_e[bid]
            stars_ne = bid_2_stars[bid]
            lestars = len(stars_e)
            if(lestars >= min_thold and lestars < max_thold):
                x = []
                x.append([np.mean(stars_ne)])
                countratings = [Counter(stars_ne)[i] for i in range(1,6)]
                distribution = [x/sum(countratings) for x in countratings]
                x.append(distribution)
                x.append(give_categ(np.mean(stars_ne)))
                x = [item for sublist in x for item in sublist]
                x1.append(x)
                y.append(np.mean(stars_e))                        
    xfeat = np.array(x1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    xfeat = imp.fit_transform(xfeat)
    yfeat = np.array(y).reshape(-1,)
    print("Dimension of Our Dataset: ", xfeat.shape)
    return xfeat,yfeat
  
  
# def create_dataset_everything(min_thold, bdata, bid_2_stars, bid_2_stars_e, bid_2_textenc, max_thold=float('inf')):
#     basicprint = False
#     x1 = [] f
#     x2 = []
#     y = []
#     bids = bdata['business_id'].values
#     for bid in bids:
#         if bid in bid_2_stars and bid in bid_2_stars_e and bid in bid_2_textenc:
#             stars_e = bid_2_stars_e[bid]
#             stars_ne = bid_2_stars[bid]
#             lestars = len(stars_e)
#             if(lestars >= min_thold and lestars < max_thold):
#                 x = []
#                 x.append([np.mean(stars_ne)])
#                 x.append(give_categ(np.mean(stars_ne)))
#                 countratings = [Counter(stars_ne)[i] for i in range(1,6)]
#                 distribution = [x/sum(countratings) for x in countratings]
#                 x.append(distribution)
#                 x = [item for sublist in x for item in sublist]
#                 x1.append(x)
#                 y.append(np.mean(stars_e))                        
#     xfeat = np.array(x1)
#     imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#     xfeat = imp.fit_transform(xfeat)
#     yfeat = np.array(y).reshape(-1,)
#     print("Dimension of Our Dataset: ", xfeat.shape)
#     return xfeat,yfeat
  

  
def create_dataset_onlytext(min_thold, bdata, bid_2_stars, bid_2_stars_e, bid_2_textenc, max_thold=float('inf')):
    basicprint = False
    x1 = []
    x2 = []
    y = []
    bids = bdata['business_id'].values
    for bid in bids:
        if bid in bid_2_stars and bid in bid_2_stars_e and bid in bid_2_textenc:
            stars_e = bid_2_stars_e[bid]
            stars_ne = bid_2_stars[bid]
            lestars = len(stars_e)
            if(lestars >= min_thold and lestars < max_thold):
                x = []
                x.append(bid_2_textenc[bid])
                x = [item for sublist in x for item in sublist]
                x1.append(x)
                y.append(np.mean(stars_e))                        
    xfeat = np.array(x1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    xfeat = imp.fit_transform(xfeat)
    yfeat = np.array(y).reshape(-1,)
    print("Dimension of Our Dataset: ", xfeat.shape)
    return xfeat,yfeat
  

def create_dataset_everything_text(min_thold, bdata, bid_2_stars, bid_2_stars_e, bid_2_textenc, max_thold=float('inf')):
    basicprint = False
    x1 = []
    x2 = []
    y = []
    bids = bdata['business_id'].values
    for bid in bids:
        if bid in bid_2_stars and bid in bid_2_stars_e and bid in bid_2_textenc:
            stars_e = bid_2_stars_e[bid]
            stars_ne = bid_2_stars[bid]
            lestars = len(stars_e)
            if(lestars >= min_thold and lestars < max_thold):
                x = []
                x.append([np.mean(stars_ne)])
                countratings = [Counter(stars_ne)[i] for i in range(1,6)]
                distribution = [x/sum(countratings) for x in countratings]
                x.append(distribution)
                x.append(give_categ(np.mean(stars_ne)))
                x.append(bid_2_textenc[bid])
                x = [item for sublist in x for item in sublist]
                x1.append(x)
                y.append(np.mean(stars_e))                        
    xfeat = np.array(x1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    xfeat = imp.fit_transform(xfeat)
    yfeat = np.array(y).reshape(-1,)
    print("Dimension of Our Dataset: ", xfeat.shape)
    return xfeat,yfeat

  
def create_dataset_stddev_avg(min_thold, bdata, bid_2_stars, bid_2_stars_e, max_thold=float('inf')):
    basicprint = False
    x1 = []
    x2 = []
    y = []
    bids = bdata['business_id'].values
    for bid in bids:
        if bid in bid_2_stars and bid in bid_2_stars_e:
            stars_e = bid_2_stars_e[bid]
            stars_ne = bid_2_stars[bid]
            lestars = len(stars_e)
            if(lestars >= min_thold and lestars < max_thold):
                x = []
                x.append([np.std(stars_ne)])
                x = [item for sublist in x for item in sublist]
                x1.append(x)
                y.append(np.std(stars_e))                        
    xfeat = np.array(x1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    xfeat = imp.fit_transform(xfeat)
    yfeat = np.array(y).reshape(-1,)
    print("Dimension of Our Dataset: ", xfeat.shape)
    return xfeat,yfeat

  
def create_dataset_everything_text_plus_len(min_thold, bdata, bid_2_stars, bid_2_stars_e, bid_2_textenc, bid_2_textlen, max_thold=float('inf')):
    basicprint = False
    x1 = []
    x2 = []
    y = []
    bids = bdata['business_id'].values
    for bid in bids:
        if bid in bid_2_stars and bid in bid_2_stars_e and bid in bid_2_textenc and bid in bid_2_textlen:
            stars_e = bid_2_stars_e[bid]
            stars_ne = bid_2_stars[bid]
            lestars = len(stars_e)
            if(lestars >= min_thold and lestars < max_thold):
                x = []
                x.append([np.mean(stars_ne)])
                countratings = [Counter(stars_ne)[i] for i in range(1,6)]
                distribution = [x/sum(countratings) for x in countratings]
                x.append(distribution)
                x.append(give_categ(np.mean(stars_ne)))
                x.append(bid_2_textlen[bid])
                x.append(bid_2_textenc[bid])
                x = [item for sublist in x for item in sublist]
                x1.append(x)
                y.append(np.mean(stars_e))                        
    xfeat = np.array(x1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    xfeat = imp.fit_transform(xfeat)
    yfeat = np.array(y).reshape(-1,)
    print("Dimension of Our Dataset: ", xfeat.shape)
    return xfeat,yfeat
  

  
def create_dataset_multiclass(min_thold, bdata, bid_2_stars, bid_2_stars_e, bid_2_textenc, max_thold=float('inf')):
    basicprint = False
    x1 = []
    x2 = []
    y = []
    bids = bdata['business_id'].values
    for bid in bids:
        if bid in bid_2_stars and bid in bid_2_stars_e and bid in bid_2_textenc:
            stars_e = bid_2_stars_e[bid]
            stars_ne = bid_2_stars[bid]
            lestars = len(stars_e)
            if(lestars >= min_thold and lestars < max_thold):
                x = []
                x.append([np.mean(stars_ne)])
                countratings_ne = [Counter(stars_ne)[i] for i in range(1,6)]
                distribution_ne = [x/sum(countratings_ne) for x in countratings_ne]
                x.append(distribution_ne)
                x.append(give_categ(np.mean(stars_ne)))
                x.append(bid_2_textenc[bid])
                x = [item for sublist in x for item in sublist]
                x1.append(x)
                countratings_e = [Counter(stars_e)[i] for i in range(1,6)]
                distribution_e = [x/sum(countratings_e) for x in countratings_e]
                y.append(distribution_e)                        
    xfeat = np.array(x1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    xfeat = imp.fit_transform(xfeat)
    yfeat = np.array(y)
    print("Dimension of Our Dataset: ", xfeat.shape)
    return xfeat,yfeat