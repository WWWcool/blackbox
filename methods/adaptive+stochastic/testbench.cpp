/*
 * File:   adaptive+stochastic_method.cpp
 * Author: andrei
 *
 * Created on January 20, 2019, 5:44 PM
 */

#include <iostream>
#include "adaptive+stochastic_method.hpp"
#include "../adaptive/adaptive_method.hpp"
#include "../best_point/best_point_method.hpp"
#include "../granular/granular_method.hpp"
#include "../../brute/bruteforce.hpp"
#include <common/bbsolver.hpp>
#include "math.h"

#include "testfuncs/manydim/benchmarks.hpp"
#include "oneobj/contboxconstr/benchmarkfunc.hpp"

using BM = Benchmark<double>;
using namespace std;

struct testRes
{
    double globMin;
    double findMin;
    std::string desc;
};

testRes execSM(
    int type,
    double mInc,
    double mDec,
    int dim,
    double* x,
    const double* a,
    const double* b,
    const std::function<double(const double*)> func
){
    if(type == 0){
        LOCSEARCH::AdaptiveStochasticMethod<double> searchMethod;
        searchMethod.getOptions().mDoTracing = false;
        searchMethod.getOptions().maxStepNumber = 100;
        searchMethod.getOptions().mInc = mInc;
        searchMethod.getOptions().mDec = mDec;
        searchMethod.getOptions().numbOfPoints = 100;
        double result = searchMethod.search(dim, x, a, b, func);

        testRes res;
        res.findMin = result;

        return res;
    }

    if(type == 1){
        LOCSEARCH::AdaptiveMethod<double> searchMethod;
        searchMethod.getOptions().mDoTracing = false;
        searchMethod.getOptions().maxStepNumber = 100;
        searchMethod.getOptions().mInc = mInc;
        searchMethod.getOptions().mDec = mDec;
        searchMethod.getOptions().numbOfPoints = 100;

        double result = searchMethod.search(dim, x, a, b, func);

        testRes res;
        res.findMin = result;

        return res;
    }

    if(type == 2){
        LOCSEARCH::BestPointMethod<double> searchMethod;
        searchMethod.getOptions().mDoTracing = false;
        searchMethod.getOptions().maxStepNumber = 100;
        searchMethod.getOptions().mDec = mDec;
        searchMethod.getOptions().numbOfPoints = 100;

        double result = searchMethod.search(dim, x, a, b, func);

        testRes res;
        res.findMin = result;

        return res;
    }

    if(type == 3){
        LOCSEARCH::GranularMethod<double> searchMethod;
        searchMethod.getOptions().mDoTracing = false;
        searchMethod.getOptions().maxStepNumber = 100;

        double result = searchMethod.search(dim, x, a, b, func);

        testRes res;
        res.findMin = result;

        return res;
    }

    if(type == 5){
        BruteForce<double> searchMethod(16);

        double result = searchMethod.search(dim, x, a, b, func);

        testRes res;
        res.findMin = result;

        return res;
    }
}

double func(const double* x) {
    return 100 * SGSQR(x[1] - x[0] * x[0]) + SGSQR(1 - x[1]);
}

testRes testBench_3(int type, std::shared_ptr<BM> bm, double mInc, double mDec) {
    const int dim = bm->getDim();

    OPTITEST::BenchmarkProblemFactory problemFactory(bm);
    COMPI::MPProblem<double> *mpp = problemFactory.getProblem();

    double a[dim], b[dim];
    double x[dim];
    for (int i = 0; i < dim; i++) {
        a[i] = bm->getBounds()[i].first ;
        b[i] = bm->getBounds()[i].second;
        x[i] = (b[i] + a[i]) / 2.0;
    }

    std::function<double (const double*) > func = [&] (const double * x) {
        return mpp->mObjectives.at(0)->func(x);
    };

    testRes res = execSM(type, mInc, mDec, dim, x, a, b, func);
    res.globMin = bm->getGlobMinY();
    res.desc = bm->getDesc();
    return res;
}

testRes testBench_2(double mInc, double mDec) {
    const int dim = 2;

    double x[dim] = {3, 3};

    double a[dim], b[dim];
    std::fill(a, a + dim, -4);
    std::fill(b, b + dim, 8);

    LOCSEARCH::AdaptiveStochasticMethod<double> searchMethod;
    searchMethod.getOptions().mDoTracing = false;
    searchMethod.getOptions().maxStepNumber = 100;
    searchMethod.getOptions().mInc = mInc;
    searchMethod.getOptions().mDec = mDec;
    searchMethod.getOptions().numbOfPoints = 100;
    double result = searchMethod.search(dim, x, a, b, func);

    testRes res;
    res.globMin = 0;
    res.findMin = result;
    res.desc = "Test func ";

    return res;
}

testRes testBench(std::shared_ptr<BM> bm, double mInc, double mDec) {
    const int dim = bm->getDim();

    OPTITEST::BenchmarkProblemFactory problemFactory(bm);
    COMPI::MPProblem<double> *mpp = problemFactory.getProblem();

    LOCSEARCH::AdaptiveStochasticMethod<double> searchMethod;
    searchMethod.getOptions().mDoTracing = false;
    searchMethod.getOptions().maxStepNumber = 100;
    searchMethod.getOptions().mInc = mInc;
    searchMethod.getOptions().mDec = mDec;
    searchMethod.getOptions().numbOfPoints = 100;

    double a[dim], b[dim];
    double x[dim];
    for (int i = 0; i < dim; i++) {
        a[i] = bm->getBounds()[i].first ;
        b[i] = bm->getBounds()[i].second;
        x[i] = (b[i] + a[i]) / 2.0;
    }

    std::function<double (const double*) > func = [&] (const double * x) {
        return mpp->mObjectives.at(0)->func(x);
    };

    double result = searchMethod.search(dim, x, a, b, func);

    testRes res;
    res.globMin = bm->getGlobMinY();
    res.findMin = result;
    res.desc = bm->getDesc();

    return res;
}

double testLayer(const double* x){
    double Avg = 0;
    double Sum = 0;

    Benchmarks<double> tests;
    double ok = 0;
    double all = 0;
    for(auto bm : tests){
        testRes res;
        all++;
        res = testBench(bm, x[0], x[1]);
        double funRes = fabs(res.globMin - res.findMin);
        if(funRes < 0.1){
            ok++;
        }
        // Sum += funRes*funRes;
        // funRes < 0.1 ? std::cout << "\033[32m.\033[0m" : std::cout << "\033[31m.\033[0m";
    }
    double percent = 100*ok/all;
    std::cout << "\033[1;33mResult:" << percent << "% \033[0m\n";
    if(percent > 90){
        if(percent > 95){
            std::cout << "\033[1;31mResult:" << percent << "- point - " << snowgoose::VecUtils::vecPrint(2, x) << " \033[0m\n";
        }
        else{
            std::cout << "\033[1;32mResult:" << percent << "- point - " << snowgoose::VecUtils::vecPrint(2, x) << " \033[0m\n";
        }

    }
    // Avg = sqrt(Sum)/all;
    return 100 - percent;
}

double testLayer_2(const double* x){
    testRes res;
    res = testBench_2(x[0], x[1]);
    double funRes = fabs(res.globMin - res.findMin);
    funRes < 0.1 ? std::cout << "\033[32m.\033[0m" : std::cout << "\033[31m.\033[0m";

    return funRes;
}

std::string getSMName(int type){
    if(type == 0){
        return "AdaptiveStochasticMethod \t";
    }

    if(type == 1){
        return "AdaptiveMethod \t\t\t";
    }

    if(type == 2){
        return "BestPointMethod \t\t";
    }

    if(type == 3){
        return "GranularMethod \t\t\t";
    }

    if(type == 5){
        return "BruteForce \t\t\t";
    }
}

int main(int argc, char** argv) {
    const int testCase = argc > 1 ? atoi(argv[1]) : 0;
    double mInc = 1.32141; // 1.1 - 2
    double mDec = 0.495813; // 0.1 - 0.9

    if(argc > 3){
        mInc = atoi(argv[2]);
        mDec = atoi(argv[3]);
    }

    if(testCase == 1){
        LOCSEARCH::AdaptiveStochasticMethod<double> searchMethod;
        searchMethod.getOptions().mDoTracing = false;
        searchMethod.getOptions().maxStepNumber = 100;
        searchMethod.getOptions().mInc = mInc;
        searchMethod.getOptions().mDec = mDec;
        searchMethod.getOptions().numbOfPoints = 1000;
        double a[2] = {1.1, 0.1}, b[2] = {2, 0.9};
        double x[2] = {1.6, 0.5};

        std::function<double (const double*) > func = [&] (const double * x) {
            // return testLayer_2(x);
            return testLayer(x);
        };

        double result = searchMethod.search(2, x, a, b, func);

        std::cout /*<< "Found value = " */<< result << "\t";
        std::cout /* << "At " */<< snowgoose::VecUtils::vecPrint(2, x) << "\t" << "\n";
    }
    else{
        Benchmarks<double> tests;
        double all = 0;

        const int SMCount = 4;
        double ok[SMCount] = {0,0,0,0};

        for(auto bm : tests){
            testRes res;
            all++;
            std::cout << bm->getDesc() << "\n";
            for(int i = 0; i < SMCount; i++){
                res = testBench_3(i, bm, mInc, mDec);

                if(fabs(res.globMin - res.findMin) < 0.1){
                    std::cout << "\033[32m" << getSMName(i);
                    ok[i]++;
                }
                else{
                    std::cout << "\033[31m" << getSMName(i);
                }

                std::cout << res.globMin << "\t";
                std::cout << res.findMin << "\t";
                std::cout << "\033[0m\n";
            }
        }
        for(int i = 0; i < SMCount; i++){
            std::cout << "\033[1;33mResult:" << 100*ok[i]/all << "% OK! \033[0m\n";
        }
    }

    return 0;
}

