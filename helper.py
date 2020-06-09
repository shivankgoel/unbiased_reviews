from dependencies import *

def give_text_feats_tfidf(review_data):
    review_data_ne = review_data.loc[review_data['iselite'] == False]
    bid_2_stars = review_data_ne.groupby('business_id')['stars'].apply(list).to_dict()
    bid_2_textrev = review_data_ne.groupby('business_id')['text'].apply(list).to_dict()
    bid_2_text = {i:' '.join(j) for i,j in bid_2_textrev.items() if all(isinstance(x, str) for x in j)}
    cv = TfidfVectorizer(min_df = 400, stop_words="english")
    textfeats = cv.fit_transform(bid_2_text.values())
    print("Size of Vocabulary: ", len(cv.vocabulary_))
    bid_2_textenc = {a:b.toarray()[0] for a,b in zip(bid_2_text.keys(),textfeats)}
    return cv, bid_2_stars, review_data_ne, bid_2_textenc
  

def give_text_feats_tfidf_elite(review_data, vocab):
    review_data_ne = review_data.loc[review_data['iselite'] == True]
    bid_2_stars = review_data_ne.groupby('business_id')['stars'].apply(list).to_dict()
    bid_2_textrev = review_data_ne.groupby('business_id')['text'].apply(list).to_dict()
    bid_2_text = {i:' '.join(j) for i,j in bid_2_textrev.items() if all(isinstance(x, str) for x in j)}
    cv = TfidfVectorizer(vocabulary = vocab, stop_words="english")
    textfeats = cv.fit_transform(bid_2_text.values())
    print("Size of Vocabulary: ", len(cv.vocabulary_))
    bid_2_textenc = {a:b.toarray()[0] for a,b in zip(bid_2_text.keys(),textfeats)}
    return cv, bid_2_stars, review_data_ne, bid_2_textenc
  
def give_text_feats_cv_nonelite(review_data):
    review_data_ne = review_data.loc[review_data['iselite'] == False]
    bid_2_stars = review_data_ne.groupby('business_id')['stars'].apply(list).to_dict()
    bid_2_textrev = review_data_ne.groupby('business_id')['text'].apply(list).to_dict()
    bid_2_text = {i:' '.join(j) for i,j in bid_2_textrev.items() if all(isinstance(x, str) for x in j)}
    cv = CountVectorizer(min_df = 400, stop_words="english")
    textfeats = cv.fit_transform(bid_2_text.values())
    print("Size of Vocabulary: ", len(cv.vocabulary_))
    bid_2_textenc = {a:b.toarray()[0] for a,b in zip(bid_2_text.keys(),textfeats)}
    return cv, bid_2_stars, review_data_ne, bid_2_textenc

  
def give_raw_text(review_data, elitedata = False):
    review_data_ne = review_data.loc[review_data['iselite'] == elitedata]
    bid_2_stars = review_data_ne.groupby('business_id')['stars'].apply(list).to_dict()
    bid_2_textrev = review_data_ne.groupby('business_id')['text'].apply(list).to_dict()
    bid_2_text = {i:' '.join(j) for i,j in bid_2_textrev.items() if all(isinstance(x, str) for x in j)}
    bids  = [k for k,v in bid_2_text.items()]
    return bids, bid_2_stars, bid_2_text


# def give_categ(avg):
#     temp = np.zeros(17)
#     c = int((avg-1)/0.25) 
#     temp[c] = 1
#     return temp


def create_dataset(min_thold, bdata, bid_2_stars, bid_2_stars_e, bid_2_textenc, textenc = False, vec = False, max_thold=float('inf')):
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
                x.append([Counter(stars_ne)[i] for i in range(1,6)])
                x.append([len(stars_ne)])
                x.append([np.mean(stars_ne)])
                if textenc:
                    if basicprint:
                        print('Num features before text: ', len([item for sublist in x for item in sublist]))
                    x.append(bid_2_textenc[bid])
                    if basicprint:
                        print('Num features after text: ', len([item for sublist in x for item in sublist]))
                        basicprint = False
                x = [item for sublist in x for item in sublist]
                x1.append(x)
                x = []
                if vec:
                    x.append(give_categ(np.mean(stars_ne)))
                x = [item for sublist in x for item in sublist]
                x2.append(x)
                y.append(np.mean(stars_e))                        
    x1 = np.array(x1)
    x2 = np.array(x2)
    if not vec:
        x2 = x2.reshape(-1,1)
    xfeat = np.concatenate((x1,x2),axis=1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    xfeat = imp.fit_transform(xfeat)
    yfeat = np.array(y).reshape(-1,)
    print("Dimension of Our Dataset: ", xfeat.shape)
    return xfeat,yfeat


  
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
                x.append(distrbution[:-1])
                x = [item for sublist in x for item in sublist]
                x1.append(x)
                y.append(np.mean(stars_e))                        
    xfeat = np.array(x1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    xfeat = imp.fit_transform(xfeat)
    yfeat = np.array(y).reshape(-1,)
    print("Dimension of Our Dataset: ", xfeat.shape)
    return xfeat,yfeat

  

def give_train_test(xfeat, yfeat):
    x_train,x_test,y_train,y_test = train_test_split(xfeat, yfeat, test_size=0.20, random_state=4, shuffle = True)
    return x_train, x_test, y_train.reshape(-1,), y_test.reshape(-1,)
  

def give_train_test_multi(xfeat, yfeat):
    x_train,x_test,y_train,y_test = train_test_split(xfeat, yfeat, test_size=0.20, random_state=4, shuffle = True)
    return x_train, x_test, y_train, y_test


def print_prediction(fpredict, y_test_actual):
    print("MSE: ", round(mean_squared_error(y_test_actual, fpredict),4))
    print("R-Squared : ", round(r2_score(y_test_actual, fpredict),4))
    x = [np.round(0.1 * x,2) for x in range(1,6) ]
    y = []
    for xt in x:
        temp = np.abs(fpredict - y_test_actual) <= xt
        y.append(np.sum(temp)/len(temp))
    print("Margin  0.10  0.20  0.30  0.40  0.50")
    print("        %.2f  %.2f  %.2f  %.2f  %.2f" %(y[0]*100,y[1]*100,y[2]*100,y[3]*100,y[4]*100))
    print(" ")
    return r2_score(y_test_actual, fpredict)
    

def print_outliers(ypred, y_test):
    print("Outliers")
    print(np.sum(ypred > y_test)/len(y_test))
    print(np.sum(ypred < y_test)/len(y_test))
    print(np.sum(ypred - y_test > 0.3)/len(y_test))
    print(np.sum(ypred - y_test < -0.3)/len(y_test))
    

def test_model(model,xtrain,ytrain,xtest,ytest, out=False):
    model.fit(xtrain,ytrain)
    rsqr = print_prediction(model.predict(xtest), ytest)
    if out:
        print_outliers(model.predict(xtest), ytest)
    return model, rsqr

    
def give_reg_model_gb(d = 3):
    print("Gradient Boost")
    model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=d,random_state=4,loss='ls')
    return model


# def give_reg_model_rf(n=100):
#     print("Random Forest")
#     x = []
#     model = RandomForestRegressor(n_estimators = n, random_state = 4)
#     return model
  
def give_reg_model_rf(n=100,mins = 1):
    print("Random Forest")
    model = RandomForestRegressor(n_estimators = n,oob_score = True,min_samples_leaf=mins,random_state=2, n_jobs=-1)
    return model
  
  
def give_reg_model_rf_pca(n=100,pcadim=125,mins=1):
    print("Random Forest")
    x = []
    x.append(('pca', PCA(pcadim)))
    x.append(('rf_regression', RandomForestRegressor(n_estimators = n, random_state = 50, min_samples_leaf  = mins,)))
    model = Pipeline(x)
    return model

def give_reg_model_nn(n, a=100, b=66):
    print("Neural Network")
    model = Sequential()
    model.add(Dense(a,input_dim=n,activation='relu',kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(rate = 0.5))
    model.add(Dense(b, activation='relu', kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def give_reg_model_svm(pcadim=125, c=1.0, eps=0.2):
    print("SVM")
    x = []
    x.append(('pca', PCA(pcadim)))
    x.append(('svm', SVR(gamma='scale', C= c, epsilon = eps)))
    model = Pipeline(x)
    return model


def give_reg_model_linear(a=0.8):
    print("Linear")
    model = linear_model.Ridge(alpha=a)
    return model


def give_reg_model_bagging(a=0.8, n = 100):
    print("Bagging")
    model = BaggingRegressor(base_estimator=linear_model.Ridge(alpha=a),n_estimators=n, random_state=0)
    return model

  
  
# def plot_ratings_dist(t):
#     countnumberofeachstar = [Counter(t)[i] for i in range(1,6)]
#     totalnumberofratings = len(t)
#     distribution = [round(x/totalnumberofratings,5) for x in countnumberofeachstar]
#     # print(countnumberofeachstar)
#     # print(totalnumberofratings)
#     print(distribution)
#     plt.bar([1,2,3,4,5],distribution, tick_label=distribution)
#     plt.show()
    
    
def plot_ratings_dist(t,xl="",yl=""):
    countnumberofeachstar = [Counter(t)[i] for i in range(1,6)]
    totalnumberofratings = len(t)
    distribution = [round(x/totalnumberofratings,5) for x in countnumberofeachstar]
    print(distribution)
    plt.bar([1,2,3,4,5],distribution, width=0.5)
    for a,b in zip([1,2,3,4,5], distribution):
        plt.text(a - 0.15, b + 0.004, str(np.round(b*100)))
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()
    

    
def give_reg_model_nn_multiclass(n, a=100, b=66):
    print("Neural Network")
    model = Sequential()
    model.add(Dense(a,input_dim=n,activation='relu'))
    model.add(Dense(b, activation='relu'))
    model.add(Dense(5,activation='softmax'))
    model.compile(loss='log_loss', optimizer='adam')
    return model
  


def give_bids(min_thold, bdata, bid_2_stars, bid_2_stars_e):
    ans = []
    bids = bdata['business_id'].values
    for bid in bids:
        if bid in bid_2_stars and bid in bid_2_stars_e and len(bid_2_stars_e[bid]) >= min_thold:
              ans.append(bid)
    return ans