import pandas as pd
import numpy as np
from collections import Counter
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList
df = pd.read_csv('../large_files/movielens-20m-dataset/rating.csv')

# make the user ids go from 0...N-1
df.userId = df.userId - 1

# create a mapping for movie ids
unique_movie_ids = set(df.movieId.values)

movie2idx = {}
count = 0

for movie_id in unique_movie_ids:
  movie2idx[movie_id] = count #dic[key] = value
  count += 1

# add them to the data frame
# takes awhile
df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)
df = df.drop(columns=['timestamp'])
print("Original dataframe size:", len(df))

N = df.userId.max() + 1 # number of users
M = df.movie_idx.max() + 1 # number of movies

user_ids_count = Counter(df.userId)
movie_ids_count = Counter(df.movie_idx)
# keep only the number below since the dataset is too large.
n = 5000
m = 1000

# Take the top n/m common items out
user_ids = [u for u, c in user_ids_count.most_common(n)]
movie_ids = [m for m, c in movie_ids_count.most_common(m)]

# make a copy for the valid dataset
df_small = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)].copy()

# now let's remake the ids so that they can be sequential
new_user_id_map = {}
i = 0
for old in user_ids:
    new_user_id_map[old] = i
    i += 1

new_movie_id_map = {}
j = 0
for old in movie_ids:
    new_movie_id_map[old] = j
    j += 1
    
# Similarly, let's now set the new id
print("Setting new ids.")
df_small.loc[:,'userId'] = df_small.apply(lambda row: new_user_id_map[row.userId], axis=1)
df_small.loc[:,'movie_idx'] = df_small.apply(lambda row: new_movie_id_map[row.movie_idx], axis=1)
print("max user id: ", df_small.userId.max())
print("max movie id: ", df_small.movie_idx.max())

# Now reset the N and M
N = df.userId.max()
M = df.movie_idx.max()

# split into train and test set
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

user2movie = {}
movie2user = {}
usermovie2rating = {}
print("Calling: update_user2movie_and_movie2user")
count = 0
def update_user2movie_and_movie2user(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: {}.".format(float(count)/cutoff))
    
    i = int(row.userId)
    j = int(row.movie_idx)
    if i not in user2movie:
        user2movie[i] = [j]
    else:
        user2movie[i].append(j)
    
    if j not in movie2user:
        movie2user[j] = [i]
    else:
        movie2user[j].append(i)
        
    usermovie2rating[(i,j)] = row.rating
df_train.apply(update_user2movie_and_movie2user, axis=1)

# test ratings dictionary
usermovie2rating_test = {}
print("Calling: update_usermovie2rating_test")
count = 0
def update_usermovie2rating_test(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: {}.".format(float(count)/cutoff))
    
    i = int(row.userId)
    j = int(row.movie_idx)
    usermovie2rating_test[(i,j)] = row.rating
df_test.apply(update_usermovie2rating_test, axis=1)


# let's start the item-item-collaborative filtering
N = np.max(list(user2movie.keys())) + 1
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u,m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N: ", N)
print("M: ", M)

K = 20 # number of neighbors we'd like to consider
limit = 5 # number of common movies users must have in common to consider valid
neighbors = [] # store neighbors in this list
averages = [] # each item's average rating for later use
deviations = [] # each item's deviation for later use

for i in range(M):
    t0 = datetime.now()
    # find the K closest items to item i
    users_i = movie2user[i]
    users_i_set = set(users_i)
    
    # calculate avg and deviation
    ratings_i = {user:usermovie2rating[(user, i)] for user in users_i}
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = {user:(rating - avg_i) for user, rating in ratings_i.items()}
    dev_i_values = np.array(list(dev_i_values.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))
    
    # save these for later use
    averages.append(avg_i)
    deviations.append(dev_i)
    
    sl = SortedList()
    
    for j in range(M):
        # don't include yourself
        if j != i:
            user_j = movie2user[j]
            user_j_set = set(user_j)
            common_users = (users_i_set & user_j_set)
            if len(common_users) > limit:
                # calculate avg and deviation
                ratings_j = {user:usermovie2rating[(user, j)] for user in user_j}
                avg_j = np.mean(ratings_i.values())
                dev_j = {user:(rating - avg_j) for user, rating in ratings_j.items()}
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))
                
                # calculate correlation coefficient
                numerator = sum(dev_i[m]*dev_j[m] for m in common_users)
                w_ij = numerator / (sigma_i * sigma_j)
                
                # negate weight, because list is sorted ascending
                sl.add((-w_ij, j))
                if len(sl) > K:
                    del sl[-1]
    
    # store the neighbors
    neighbors.append(sl)
    
    # print out useful things
    if i % 1 == 0:
        print(i)
        print("Time: {}".format(datetime.now() - t0))

def predict(i, u):
    numerator = 0
    denominator = 0
    for neg_w, j in neighbors[i]:
        try:
            numerator += -neg_w * deviations[j][u]
            denominator += abs(neg_w)
        except KeyError:
            pass
    
    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]
    prediction = min(5, prediction)
    prediction = max(0.5, prediction)
    
    return(prediction)

train_predictions = []
train_targets = []
for (u,m), target in usermovie2rating.items():
    prediction = predict(m,u)
    
    train_predictions.append(prediction)
    train_targets.append(prediction)

test_predictions = []
test_targets = []
for (u,m),target in usermovie2rating_test.items():
    prediction = predict(m,u)
    
    test_predictions.append(prediction)
    test_targets.append(target)

def rmse(p, t):
    p = np.array(p)
    t = np.array(t)
    return(np.mean(np.sqrt((p-t)**2)))

print('train rmse: ', rmse(train_predictions, train_targets))
print('test rmse: ', rmse(test_predictions, test_targets))