//#include "cycletimer.h"
#include <iostream>
#include <string>
#include <math.h>
#include <fstream>
#define MAX_RATINGS 110000
#define MAX_MOVIES 110000
#define MAX_CUSTOMERS 110000
#define MAX_FEATURES 10
#define MIN_EPOCHS 10

#define MIN_IMPROVEMENT 0.0001
#define INIT 0.1
#define LRATE 0.001
#define K 0.015

using namespace std;
const string TRAIN_PATH = "../ml-100k/u1.base";

typedef struct Movie {
  int rating_count;
  int rating_sum;
  double rating_mean;
} Movie;

typedef struct Customer {
  int id;
  int rating_count;
  int rating_sum;
} Customer;

typedef struct Data {
  int cust_id;
  int mov_id;
  int rating;
  float cache;
} Data;

class Recommender {
private:
  int rating_count;
  Data ratings[MAX_RATINGS];
  Movie movies[MAX_MOVIES];
  Customer customers[MAX_CUSTOMERS];
  float U[MAX_FEATURES][MAX_MOVIES];
  float I[MAX_FEATURES][MAX_MOVIES];
public:
  Recommender(void) {
    rating_count = 0;
  };
  ~Recommender(void){};
  void read_file();
  void preprocess();
  void calc_features();
  double predict(int mov_id, int cust_id, int feature, float cache, bool bTrailing);
};

int main(int argc, char *argv[]) {
  Recommender *recommender = new Recommender();
  recommender->read_file();
}

void Recommender::read_file() {
  //double start = CycleTimer::currentSeconds();
  int user, movie, rating, timestamp;
  ifstream in_file(TRAIN_PATH);
  while (in_file >> user >> movie >> rating >> timestamp) {
    ratings[rating_count].cust_id = user;
    ratings[rating_count].mov_id = movie;
    ratings[rating_count].rating = rating;
    ratings[rating_count].cache = 0;
    rating_count++;
  }
  //double end = CycleTimer::currentSeconds();
  //cout << "Reading file takes " << end - start << " seconds!" << endl;
}

void Recommender::preprocess() {
  for (int i = 0; i < rating_count; i++) {
    // populate movies
    movies[ratings[i].mov_id].rating_count++;
    movies[ratings[i].mov_id].rating_sum += ratings[i].rating;
    // populate customers
    customers[ratings[i].cust_id].rating_count++;
    customers[ratings[i].cust_id].rating_sum += ratings[i].rating;
  }
  // calculat movie means
  for (int i = 0; i < MAX_MOVIES; i++) {
    movies[i].rating_mean = movies[i].rating_sum / (1.0 * movies[i].rating_count);
  }
}

double Recommender::predict(int mov_id, int cust_id, int feature, float cache, bool bTrailing) {
  double sum = (cache > 0) ? cache : 1;
  sum += U[feature][mov_id] * I[feature][cust_id];
  if (sum > 5) sum = 5;
  if (sum < 1) sum = 1;
  if (bTrailing) {
    sum += (MAX_FEATURES - feature - 1) * (INIT * INIT);
    if (sum > 5) sum = 5;
    if (sum < 1) sum = 1;
  }
  return sum;
}


void Recommender::calc_features() {
    int f, e, i, custId, cnt = 0;
    Data* rating;
    double err, p, sq, rmse_last, rmse = 2.0;
    short movieId;
    float cf, mf;
    for (f = 0; f < MAX_FEATURES; f++) { // train each feature
      for (e=0; (e<MIN_EPOCHS) || (rmse <= rmse_last - MIN_IMPROVEMENT); e++) {
        cnt++;
        sq = 0;
        rmse_last = rmse;
        for (i = 0; i < rating_count; i++) {
          rating = ratings + i;
          movieId = rating->mov_id;
          custId = rating->cust_id;

          p = predict(movieId, custId, f, rating->cache, true);
          err = (1.0 * rating->rating - p);
          sq += err*err;

          cf = U[f][custId];
          mf = I[f][movieId];

          // cross train
          U[f][custId] += (float)(LRATE * (err * mf - K * cf));
          I[f][movieId] += (float)(LRATE * (err * cf - K * mf));
        }
        rmse = sqrt(sq/rating_count);
      }
      for (i = 0; i < rating_count; i++) {
        rating = ratings + i;
        rating->cache = (double) predict(rating->mov_id, rating->cust_id, f, rating->cache, false);
      }
    }
}
