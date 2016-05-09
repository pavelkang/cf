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
#include "rec.h"

using namespace std;

const string TRAIN_PATH = "1.dat";
//const string TRAIN_PATH = "ml-100k/u1.base";

int compact_data[DATA_SIZE];
int compact_index[USER_SIZE];

template <typename T1, typename T2>
struct more_second {
    typedef pair<T1, T2> type;
    bool operator ()(type const& a, type const& b) const {
        return a.second > b.second;
    }
};

void CUDA_populate_user_sim_vec(int target_user, int *compact_data,
                                int *compact_index, double *sim, int topn);

void CUDA_populate_item_like_vec(int user, int *compact_data,
                                 int *compact_index, double *sim,
                                 double *like, int topn);

static void read_file_compact() {
  int user, item, rating, timestamp;
  ifstream in_file(TRAIN_PATH);
  string line;
  int num_data_read = 0;
  int num_user_read = 0;
  while (in_file >> user >> item >> rating >> timestamp) {
    if (user > num_user_read) { // new user
      compact_index[num_user_read] = 2 * num_data_read;
      num_user_read++;
      compact_data[num_data_read * 2] = item;
      compact_data[num_data_read * 2 + 1] = rating;
      num_data_read++;
    } else { // old user
      compact_data[num_data_read * 2] = item;
      compact_data[num_data_read * 2 + 1] = rating;
      num_data_read++;
    }
  }
  cout << "read "<< num_data_read << " data." << endl;
  // for (int i = 0; i < 5; i++) {
  //   cout << "user " << i+1 << "starts at " << compact_index[i] << endl;
  //   for (int j = compact_index[i]; j < compact_index[i+1]; j+=2) {
  //     cout << compact_data[j] << " - " << compact_data[j+1] << endl;
  //   }
  // }
}


static void recommend(int user, int topn = 5) {
  //pair<int, double> like_copy[ITEM_COUNT];
  double like[ITEM_SIZE];
  double sim[USER_SIZE];
  memset(like, 0, sizeof(double) * ITEM_SIZE);
  memset(sim , 0, sizeof(double) * USER_SIZE);
  // CUDA_populate_user_sim_vec(user, compact_data, compact_index, sim, topn);
  CUDA_populate_item_like_vec(user, compact_data, compact_index,
                              sim, like, topn);
  //     for (int j = 0; j < ITEM_COUNT; j++) {
  //         if (users[IDX(user,j)] == 0) {
  //             double item_score = score * users[IDX(i,j)];
  //             like[j] += item_score;
  //         }
  //     }
  // }
  // CUDA_get_recommendations(users, user, like, topn);

  // for (int i = 0; i < ITEM_COUNT; i++) {
  //     like_copy[i] = make_pair(i,like[i]);
  // }



  // sort like_table
  // output the first topn items
  // sort(std::begin(like_copy), std::end(like_copy), more_second<int, double>());
  /*
  cout << "We recommend:" << endl;
  for (int i = 0; i < topn; i++) {
    cout << like_copy[i].first << " with confidence " <<
      like_copy[i].second << endl;
  }
  */
}

int main(int argc, char *argv[]) {
  cout << "train: " << TRAIN_PATH << endl;
  cout << USER_SIZE << endl;
  double start = CycleTimer::currentSeconds();
  read_file_compact();
  double end = CycleTimer::currentSeconds();
  cout << "Reading file takes " << end - start << " seconds" << endl;
  start = CycleTimer::currentSeconds();
  //recommend(7);
  for (int i = 1; i < USER_SIZE; i++) {
    recommend(i);
  }
  end = CycleTimer::currentSeconds();
  cout << "940 recommendations took: " << end - start << " seconds" << endl;
  return 0;
}
