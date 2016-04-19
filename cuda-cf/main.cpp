#include <cstdio>
#include <iostream>
#include <string>
#include <limits.h>
//#include "cycletimer.h"
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

const string TRAIN_PATH = "../ml-100k/u1.base";
const string TEST_PATH  = "ml-100k/u1.test";

/*
typedef struct User {
  //unordered_map<int, int> ratings;
  int ratings[1500];
} User;
*/

int users[USER_COUNT][ITEM_COUNT];
int num_users;
set<int> items;

template <typename T1, typename T2>
struct more_second {
    typedef pair<T1, T2> type;
    bool operator ()(type const& a, type const& b) const {
        return a.second > b.second;
    }
};

static void read_file() {
  int user, item, rating, timestamp;
  ifstream in_file(TRAIN_PATH);
  string line;
  while (in_file >> user >> item >> rating >> timestamp) {
    items.insert(item);
    users[user][item] = rating;
  }
}

static double calc_similarity(int u, int v) {
  double u_mean = 0.0;
  double v_mean = 0.0;
  int u_size = 0;
  int v_size = 0;
  /*
  for (unordered_map<int, int>::iterator it = u->ratings.begin();
       it != u->ratings.end(); ++it) {
    u_mean += it->second;
  }

  for (unordered_map<int, int>::iterator it = v->ratings.begin();
       it != v->ratings.end(); ++it) {
    v_mean += it->second;
  }
  */

  for (int i = 0; i < ITEM_COUNT; i++) {
      int rate = users[u][i];
      u_mean += rate;
      if (rate) u_size++;
  }

  for (int j = 0; j < ITEM_COUNT; j++) {
      int rate = users[v][j];
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
  /*
  for (unordered_map<int, int>::iterator it = u->ratings.begin();
       it != u->ratings.end();
       ++it) {
    int item = it->first;
    if (v->ratings.find(item) != v->ratings.end()) {
      commons++;
      double rui = u->ratings[item];
      double rvi = v->ratings[item];
      a += (rui - u_mean)*(rvi - v_mean);
      b += (rui-u_mean)*(rui-u_mean);
      c += (rvi-v_mean)*(rvi-v_mean);
    }
  }
  */

  for (int i = 0; i < ITEM_COUNT; i++) {
      if ((users[u][i] != 0) && (users[v][i]) != 0) {
          commons++;
          double rui = users[u][i];
          double rvi = users[v][i];
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


  /*
  for (int i = 1; i < user; i++) {
    User *other_user = users[i];
    assert(other_user != NULL);
    double sim = calc_similarity(other_user, me);
    assert(!isnan(sim));
    similarity_table[other_user] = sim;
  }
  for (int i = user+1; i <= num_users; i++) {
    User *other_user = users[i];
    assert(other_user != NULL);
    double sim = calc_similarity(other_user, me);
    assert(!isnan(sim));
    similarity_table[other_user] = sim;
  }
  */
  /*
  for (map<User *, double>::iterator it = similarity_table.begin();
       it != similarity_table.end(); ++it) {
    pair<User *, double> p(it->first, it->second);
    mapcopy.push_back(p);
  }
  sort(mapcopy.begin(), mapcopy.end(), more_second<User *, double>());
  */
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
  memset(sim , 0, sizeof(double) * USER_COUNT);
  get_similar_users(user, sim, topn);
  /*
  User *me = users[user];
  assert(me != NULL);
  for (auto up : mapcopy) {
    User *other = up.first;
    double score = up.second;
    assert(!(isnan(score)));
    for (unordered_map<int, int>::iterator it = other->ratings.begin();
         it != other->ratings.end();
         ++it) {
      int item = it->first;
      if (me->ratings.find(item) == me->ratings.end()) { // i have not rated
        // add to like_table
        double item_score = score * it->second;
        assert(!(isnan(item_score)));
        if (like_table.find(item) != like_table.end()) {
          like_table[item] += item_score;
        } else {
          like_table[item] = item_score;
        }
      }
    }
  }
  */

  for (int i = 0; i < USER_COUNT; i++) {
      double score = sim[i];

      for (int j = 0; j < ITEM_COUNT; j++) {
          if (users[user][j] == 0) {
              double item_score = score * users[i][j];
              like[j] += item_score;
          }
      }
  }

  for (int i = 0; i < ITEM_COUNT; i++) {
      like_copy[i] = make_pair(i,like[i]);
  }



  /*
  // sort like_table
  vector< pair<int, double> > like_copy(like_table.begin(), like_table.end());
  */
  // output the first topn items
  sort(std::begin(like_copy), std::end(like_copy), more_second<int, double>());
  cout << "We recommend:" << endl;
  for (int i = 0; i < topn; i++) {
    cout << like_copy[i].first << " with confidence " <<
      like_copy[i].second << endl;
  }
}

int main(int argc, char *argv[]) {
  memset(users, 0, sizeof(int) * USER_COUNT * ITEM_COUNT);
  num_users = 0;
  // cout << "train: " << TRAIN_PATH << ", test: " << TEST_PATH << endl;
  // double start = CycleTimer::currentSeconds();
  read_file();
  // double end = CycleTimer::currentSeconds();
  // cout << "Reading file takes " << end - start << " seconds" << endl;
   cout << "There are " << num_users << " users, and " << items.size()
        << " items." << endl;
  recommend(7);
  return 0;
}
