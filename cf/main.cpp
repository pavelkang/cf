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
using namespace std;

const string TRAIN_PATH = "ml-100k/u1.base";
const string TEST_PATH  = "ml-100k/u1.test";

typedef struct User {
  int userid;
  unordered_map<int, int> ratings;
} User;

User *users[INT_MAX];
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
    if (users[user] == NULL) { // non-existent user
      User *u = new User();
      u->userid = user;
      u->ratings[item] = rating;
      users[user] = u;
      num_users++;
    } else {
      User *u= users[user];
      u->ratings[item] = rating;
    }
  }
}

static double calc_similarity(User *u, User *v) {
  double u_mean = 0.0;
  double v_mean = 0.0;
  for (unordered_map<int, int>::iterator it = u->ratings.begin();
       it != u->ratings.end(); ++it) {
    u_mean += it->second;
  }
  for (unordered_map<int, int>::iterator it = v->ratings.begin();
       it != v->ratings.end(); ++it) {
    v_mean += it->second;
  }
  u_mean /= u->ratings.size();
  v_mean /= v->ratings.size();
  assert(u->ratings.size() != 0);
  assert(v->ratings.size() != 0);
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  int commons = 0;
  /* Pearson correlation */
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

static void get_similar_users(int user, vector< pair<User *, double> > &mapcopy, int topn = 5) {
  map<User *, double> similarity_table; // <user, similarity>, ordered table
  User *me = users[user];
  assert(me != NULL);
  for (int i = 1; i < user; i++) {
    User *other_user = users[i];
    assert(other_user != NULL);
    double sim = calc_similarity(other_user, me);
    //assert(!isnan(sim));
    similarity_table[other_user] = sim;
  }
  for (int i = user+1; i <= num_users; i++) {
    User *other_user = users[i];
    assert(other_user != NULL);
    double sim = calc_similarity(other_user, me);
    //assert(!isnan(sim));
    similarity_table[other_user] = sim;
  }
  for (map<User *, double>::iterator it = similarity_table.begin();
       it != similarity_table.end(); ++it) {
    pair<User *, double> p(it->first, it->second);
    mapcopy.push_back(p);
  }
  sort(mapcopy.begin(), mapcopy.end(), more_second<User *, double>());
}

static void recommend(int user, int topn = 5) {
  map<int, double> like_table;
  vector< pair<User *, double> > mapcopy;
  get_similar_users(user, mapcopy, topn);
  for (vector< pair<User *, double> >::iterator it = mapcopy.begin();
       it != mapcopy.end(); it++) {
    cout << "sim " << it->first->userid << " - " << it->second << endl;
  }
  User *me = users[user];
  assert(me != NULL);
  for (auto up : mapcopy) {
    User *other = up.first;
    double score = up.second;
    //assert(!(isnan(score)));
    for (unordered_map<int, int>::iterator it = other->ratings.begin();
         it != other->ratings.end();
         ++it) {
      int item = it->first;
      if (me->ratings.find(item) == me->ratings.end()) { // i have not rated
        // add to like_table
        double item_score = score * it->second;
        //assert(!(isnan(item_score)));
        if (like_table.find(item) != like_table.end()) {
          like_table[item] += item_score;
        } else {
          like_table[item] = item_score;
        }
      }
    }
  }
  // sort like_table
  vector< pair<int, double> > like_copy(like_table.begin(), like_table.end());
  // output the first topn items
  sort(like_copy.begin(), like_copy.end(), more_second<int, double>());
  cout << "We recommend:" << endl;
  for (int i = 0; i < topn; i++) {
    cout << like_copy[i].first << " with confidence " <<
      like_copy[i].second << endl;
  }
}

int main(int argc, char *argv[]) {
  memset(users, 0, INT_MAX);
  num_users = 0;
  // cout << "train: " << TRAIN_PATH << ", test: " << TEST_PATH << endl;
  // double start = CycleTimer::currentSeconds();
  read_file();
  // double end = CycleTimer::currentSeconds();
  // cout << "Reading file takes " << end - start << " seconds" << endl;
  // cout << "There are " << num_users << " users, and " << items.size()
  //      << " items." << endl;
  for (int i = 1; i <= 940; i++) {
    recommend(i);
  }
  return 0;
}
