#include <iostream>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include "shared/myFunc.hpp"
#include "shared/Map.hpp"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/matplotlibcpp.h"
namespace plt = matplotlibcpp;
bool isLaminar(Eigen::VectorXd state, double epsilon);
int main(){
    auto start = std::chrono::system_clock::now();
    PGparams params;
    params.a = 1.8;
    params.a_prime = 1.7;
    params.omega = 0.225;
    long long n = 100000;
    long long dump = 0;
    Eigen::VectorXd x_0 = npy2EigenVec<double>("../initials/a1.8-1.7_omega0.225_n100.npy", true);
    double epsilon = 0.1;
    double check = 100;
    double progress = 10;
    int perturb_min = -15;
    int perturb_max = -8;
    int limit = 1e+8; //limitation of trial of stagger and step
    int numThreads = omp_get_max_threads();

    PGMap PG(params, n, dump, x_0);
    Eigen::MatrixXd calced_laminar = myfunc::SaS_of_map(PG, isLaminar, epsilon, progress, check, perturb_min, perturb_max, limit, numThreads);
    std::cout << "calced_laminar.cols() = " << calced_laminar.cols() << std::endl;
    /*
            █             
    █████   █          █
    ██  ██  █          █
    ██   █  █   ████  ████
    ██  ██  █  ██  ██  █  
    █████   █  █    █  █  
    ██      █  █    █  █  
    ██      █  █    █  █  
    ██      █  ██  ██  ██ 
    ██      █   ████    ██
    */
    std::cout << "plotting" << std::endl;
    // plot settings
    int skip = 1; // plot every skip points
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1200, 1200);
    std::vector<double> x(calced_laminar.cols()),y(calced_laminar.cols());

    for(int i=0;i<calced_laminar.cols();i++){
        x[i]=calced_laminar(0, i);
        y[i]=calced_laminar(1, i);
    }
    plt::scatter(x,y);

    std::ostringstream oss;
    oss << "../../generated_lam_img/gen_laminar_a" << params.a << "-" << params.a_prime << "_omega" << params.omega << "_n" << n << "_check" << check << "_progress" << progress << "_perturb" << perturb_min << "-" << perturb_max << ".png";
    std::string filename = oss.str(); // 文字列を取得する
    if (calced_laminar.cols() > 1){
        std::cout << "\n Saving result to " << filename << std::endl;
        plt::save(filename);
    }

    /*
      ██████
     ███ ███
    ██                                             █
    ██                                             █
    ██          █████   ██      █   █████        ██████    █████          █ █████     █ █████   ██      █
     ██        ██  ███   █     ██  ██   ██         ██     ███  ██         ███  ███    ███  ███   █     ██
      ███           ██   ██    ██  █     █         █     ██     ██        ██    ██    ██    ██   ██    ██
        ███         ██   ██   ██  ██     ██        █     ██      █        █     ██    █      █   ██    █
          ██    ██████    █   ██  █████████        █     ██      █        █      █    █      ██   █   ██
           ██ ███   ██    ██  █   ██               █     ██      █        █      █    █      ██   ██  ██
           ██ ██    ██    ██ ██   ██               █     ██      █        █      █    █      █     █  █
           █  ██    ██     █ ██   ██               █      █     ██        █      █    ██    ██     █ ██
    ███  ███  ██   ███     ███     ███  ██         ██     ███  ██         █      █    ███  ███     ███
    ██████     ████  █      ██      ██████          ███    █████          █      █    █ █████       ██
                                                                                      █             ██
                                                                                      █             █
                                                                                      █            ██
                                                                                      █           ██
                                                                                      █         ███
    */

    oss.str("");
    if(calced_laminar.cols() > 1){
        if (progress == n){
            oss << "../../initials/a" << params.a << "-" << params.a_prime << "_omega" << params.omega << "_n" << n << ".npy";
            std::string fname = oss.str(); // 文字列を取得する
            std::cout << "saving as " << fname << std::endl;
            EigenVec2npy(calced_laminar.col(0), fname);
        } else{
            oss << "../../generated_lam/gen_laminar_a" << params.a << "-" << params.a_prime << "_omega" << params.omega << "_n" << n << "_check" << check << "_progress" << progress << "_perturb" << perturb_min << "-" << perturb_max << ".npy";
            std::string fname = oss.str(); // 文字列を取得する
            std::cout << "saving as " << fname << std::endl;
            EigenMat2npy(calced_laminar, fname);
        }
    }
    myfunc::duration(start);
}

bool isLaminar(Eigen::VectorXd state, double epsilon){
    if (std::abs(state(0) - state(1)) < epsilon){
        return true;
    }
    else{
        return false;
    }
}