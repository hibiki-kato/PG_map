#include <iostream>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include "shared/myFunc.hpp"
#include "shared/Map.hpp"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/matplotlibcpp.h"
namespace plt = matplotlibcpp;
int main(){
    auto start = std::chrono::system_clock::now();
    PGparams params;
    params.a = 1.8;
    params.a_prime = 1.7;
    params.omega = 0.15;
    long long n = 1e+5;
    long long dump = 1e+3;
    Eigen::VectorXd x_0(2);
    x_0 << 0.4534, 0.531;

    PGMap PG(params, n, dump, x_0);
    Eigen::MatrixXd trajectory = PG.get_trajectory();
    std::vector<double> x(n+1), y(n+1);
    for (int i = 0; i < n+1; i++){
        x[i] = trajectory(0, i);
        y[i] = trajectory(1, i);
    }
    plt::figure_size(1000, 1000);
    plt::scatter(x, y);
    plt::xlim(0, 1);
    plt::ylim(0, 1);
    std::ostringstream oss;
    oss << "../../traj_img/a" << params.a << "-" << params.a_prime << "_omega" << params.omega << "_n" << n << ".png";
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "saving image to" << plotfname <<std::endl;
    plt::save(plotfname);

    oss.str("");
    oss << "../../traj/a" << params.a << "-" << params.a_prime << "_omega" << params.omega << "_n" << n << ".npy";
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "saving result to" << fname <<std::endl;
    EigenMat2npy(trajectory, fname);
    
    myfunc::duration(start);
}