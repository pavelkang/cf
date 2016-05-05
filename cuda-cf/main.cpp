#include <cstdio>
#include <iostream>
#include <string>
#include <limits.h>
#include "cycletimer.h"
#include <unordered_map>
#include <math.h>
#include <unordered_set>
#include <assert.h>
#include <fstream>
#include <set>
#include <map>
#include <algorithm>
#include <vector>
#include <cstring>

using namespace std;

#define USER_COUNT 944
#define ITEM_COUNT 2000
#define IDX(u,i) ((u) * ITEM_COUNT + i)
#define IDX_T(u,i) ((i) * USER_COUNT + u)

const string TRAIN_PATH = "../ml-100k/u1.base";
const string TEST_PATH  = "ml-100k/u1.test";

/*
typedef struct User {
  //unordered_map<int, int> ratings;
  int ratings[1500];
} User;
*/

double users[USER_COUNT * ITEM_COUNT];
double users_t[USER_COUNT * ITEM_COUNT];
double simMatrix[USER_COUNT * USER_COUNT];
int num_users;
set<int> items;

template <typename T1, typename T2>
struct more_second {
    typedef pair<T1, T2> type;
    bool operator ()(type const& a, type const& b) const {
        return a.second > b.second;
    }
};

void CUDA_get_similar_users(double *um, int user, double *similarity_copy, int topn = 5);
void CUDA_get_recommendations(double *um, int user, double *similarity_copy, int topn = 5);
void computeCorrelation(double *rate, double *rate_t, double *out) ;
void CUDA_get_rec_with_sim(double *users, double *sim, int user, double *recommendation_copy, int topn = 5);

static void read_file() {
  int user, item, rating, timestamp;
  ifstream in_file(TRAIN_PATH);
  string line;
  while (in_file >> user >> item >> rating >> timestamp) {
    items.insert(item);
    users[IDX(user,item)] = rating;
    users_t[IDX_T(user,item)] = rating;
  }
}

static double calc_similarity(int u, int v) {
  double u_mean = 0.0;
  double v_mean = 0.0;
  int u_size = 0;
  int v_size = 0;

  for (int i = 0; i < ITEM_COUNT; i++) {
      int rate = users[IDX(u,i)];
      u_mean += rate;
      if (rate) u_size++;
  }

  for (int j = 0; j < ITEM_COUNT; j++) {
      int rate = users[IDX(v,j)];
      v_mean += rate;
      if (rate) v_size++;
  }
  u_mean /= u_size;
  v_mean /= v_size;
  //assert(u->ratings.size() != 0);
  //assert(v->ratings.size() != 0);
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  int commons = 0;
  /* Pearson correlation */
  for (int i = 0; i < ITEM_COUNT; i++) {
      if ((users[IDX(u,i)] != 0) && (users[IDX(v,i)]) != 0) {
          commons++;
          double rui = users[IDX(u,i)];
          double rvi = users[IDX(v,i)];
          a += (rui - u_mean)*(rvi - v_mean);
          b += (rui-u_mean)*(rui-u_mean);
          c += (rvi-v_mean)*(rvi-v_mean);
      }
  }

  double answer;
  if (b*c == 0) {
    answer = a;
  } else {
    answer = a / (sqrt(b) * sqrt(c));
  }
  // fix pearson
  if (commons < 5) {
    answer *= 0.2 * commons;
  }
  return answer;
}


static void get_similar_users(int user, double *similarity_copy, int topn = 5) {
  //map<User *, double> similarity_table; // <user, similarity>, ordered table
  double similarity[USER_COUNT];

  for (int i = 0; i < USER_COUNT; i++) {
      if (i == user) {
          similarity[i] = 0; // exclude self
      } else {
          similarity[i] = calc_similarity(i,user);
      }
  }

  memcpy(similarity_copy, similarity, sizeof(double) * USER_COUNT);
}


static void recommend(int user, int topn = 5) {
  //map<int, double> like_table;
  //vector< pair<User *, double> > mapcopy;
  pair<int, double> like_copy[ITEM_COUNT];
  double like[ITEM_COUNT];
  double sim[USER_COUNT];
  memset(like, 0, sizeof(double) * ITEM_COUNT);
  memset(sim , 0, sizeof(double) * USER_COUNT);
  /*
  get_similar_users(user, sim, topn);
  // CUDA_get_similar_users(users, user, sim, topn);

  for (int i = 0; i < USER_COUNT; i++) {
      double score = sim[i];
      //if (i < 100) {
          //cout << "Score: " << score << endl;
      //}

      for (int j = 0; j < ITEM_COUNT; j++) {
          if (users[IDX(user,j)] == 0) {
              double item_score = score * users[IDX(i,j)];
              like[j] += item_score;
          }
      }
  }
  */
  CUDA_get_recommendations(users, user, like, topn);
  //CUDA_get_rec_with_sim(users, simMatrix, user, like, topn);

  for (int i = 0; i < ITEM_COUNT; i++) {
      like_copy[i] = make_pair(i,like[i]);
  }



  // sort like_table
  // output the first topn items
  sort(std::begin(like_copy), std::end(like_copy), more_second<int, double>());

  /*
  cout << "We recommend:" << endl;
  for (int i = 0; i < topn; i++) {
    cout << like_copy[i].first << " with confidence " <<
      like_copy[i].second << endl;
  }
  */
}

int main(int argc, char *argv[]) {
  memset(users, 0, sizeof(int) * USER_COUNT * ITEM_COUNT);
  num_users = 0;
  // cout << "train: " << TRAIN_PATH << ", test: " << TEST_PATH << endl;
  double start = CycleTimer::currentSeconds();
  read_file();
  double end = CycleTimer::currentSeconds();
  double sim[USER_COUNT];
  CUDA_get_similar_users(users, 7, sim, 5);
  cout << "Reading file takes " << end - start << " seconds" << endl;
  start = CycleTimer::currentSeconds();
  computeCorrelation(users, users_t, simMatrix);
  // cout << "There are " << num_users << " users, and " << items.size()
  //      << " items." << endl;
  //
  //recommend(7);
  for (int i = 1; i < 942; i++) {
    recommend(i);
  }
  //

  end = CycleTimer::currentSeconds();
  cout << "940 recommendations took: " << end - start << " seconds" << endl;
  return 0;
}
