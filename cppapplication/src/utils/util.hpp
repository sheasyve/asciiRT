#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <queue>
#include <stack>
#include <chrono>
#include <random>
#include <bitset>
#include <deque>
#include <tuple>
#include <numeric>
#include <climits>
#include <cassert>
#include <iterator>
#include <fstream>
#include <optional>
#include <memory>
#include <Eigen/Dense>
#include <variant>
#include <chrono>

#define _USE_MATH_DEFINES
#define ll long long
#define ull unsigned long long
#define pb push_back
#define all(x) (x).begin(), (x).end()
#define rall(v) v.rbegin(), v.rend()

const double EPS = 0.00001;;

using namespace Eigen;//Eigen library for linear algebra, matrix handling

template <typename T>
std::vector<T> vector_in(int n) {
    std::vector<T> v(n);
    for (int i = 0; i < n; i++) {
        std::cin >> v[i];
    }
    return v;
}

template <typename T>
void vector_out(const std::vector<T> &v) {
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << '\n';
}

#endif  // UTIL_HPP