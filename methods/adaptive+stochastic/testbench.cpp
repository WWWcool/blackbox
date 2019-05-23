/*
 * File:   adaptive+stochastic_method.cpp
 * Author: andrei
 *
 * Created on January 20, 2019, 5:44 PM
 */

#include <iostream>
#include "adaptive+stochastic_method.hpp"
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

double func(const double* x) {
    return 100 * SGSQR(x[1] - x[0] * x[0]) + SGSQR(1 - x[1]);
}

testRes testBench_2(double mInc, double mDec) {
    const int dim = 2;

    double x[dim] = {3, 3};

    double a[dim], b[dim];
    std::fill(a, a + dim, -4);
    std::fill(b, b + dim, 8);

    LOCSEARCH::AdaptiveStochasticMethod<double> searchMethod;
    searchMethod.getOptions().mDoTracing = false;
    searchMethod.getOptions().maxStepNumber = 50;
    searchMethod.getOptions().mInc = mInc;
    //searchMethod.getOptions().mDec = 0.368;
    searchMethod.getOptions().mDec = mDec;
    searchMethod.getOptions().numbOfPoints = 1000;
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
    searchMethod.getOptions().maxStepNumber = 50;
    searchMethod.getOptions().mInc = mInc;
    //searchMethod.getOptions().mDec = 0.368;
    searchMethod.getOptions().mDec = mDec;
    searchMethod.getOptions().numbOfPoints = 1000;

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

    // std::cout << bm->getDesc() << "\t";
    // std::cout /*<< "Glob. min. = " */<< bm->getGlobMinY() << "\t";
    // std::cout /*<< "Glob. min. x = " */ << snowgoose::VecUtils::vecPrint(dim, bm->getGlobMinX().data()) << "\t";
    // std::cout /*<< "Found value = " */<< result << "\t";
    // std::cout /* << "At " */<< snowgoose::VecUtils::vecPrint(dim, x) << "\t" << "\n";

    // std::cout /*<< "Iterations = " */<< searchMethod.getIterationsCount() << "\t";
    // std::cout /*<< "Fun. Calls count = " */<< searchMethod.getFunctionCallsCount() << "\t" << "\n";
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
    for(auto bm : tests){
        testRes res;
        res = testBench(bm, x[0], x[1]);
        // std::cout << res.desc << "\n\t\t";
        // std::cout << res.globMin << "\t";
        // std::cout << res.findMin << "\t" << "\n";
        if(fabs(res.globMin - res.findMin) < 0.1){
            Sum += (res.globMin - res.findMin)*(res.globMin - res.findMin);
        }
    }
    Avg = sqrt(Sum);
    // std::cout << "Avg = " << Avg << "\t" << "\n";
    return Avg;
}

double testLayer_2(const double* x){
    testRes res;
    res = testBench_2(x[0], x[1]);
    return fabs(res.globMin - res.findMin);
}

int main(int argc, char** argv) {
    // const double mInc = 1.418; // 1.1 - 2
    // const double mDec = 0.668; // 0.1 - 0.9

    // Benchmarks<double> tests;
    // for(auto bm : tests){
    //     testRes res;
    //     res = testBench(bm, mInc, mDec);
    //     std::cout << res.desc << "\n\t\t";
    //     std::cout << res.globMin << "\t";
    //     std::cout << res.findMin << "\t" << "\n";
    // }

    LOCSEARCH::AdaptiveStochasticMethod<double> searchMethod;
    searchMethod.getOptions().mDoTracing = false;
    searchMethod.getOptions().maxStepNumber = 50;
    searchMethod.getOptions().mInc = 1.418;
    //searchMethod.getOptions().mDec = 0.368;
    searchMethod.getOptions().mDec = 0.668;
    searchMethod.getOptions().numbOfPoints = 100;
    double a[2] = {1.1, 0.1}, b[2] = {2, 0.9};
    double x[2] = {1.6, 0.5};

    std::function<double (const double*) > func = [&] (const double * x) {
        return testLayer_2(x);
        // return testLayer(x);
    };

    double result = searchMethod.search(2, x, a, b, func);

    std::cout /*<< "Found value = " */<< result << "\t";
    std::cout /* << "At " */<< snowgoose::VecUtils::vecPrint(2, x) << "\t" << "\n";

    return 0;
}

