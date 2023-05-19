#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
#include <boost/thread.hpp>
#include <boost/atomic/atomic_ref.hpp>
#include <boost/timer/timer.hpp>
#include <mpi.h>
#include <string>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/defer.hpp>
#include <vector>
#include <boost/math/distributions/students_t.hpp>
#include "boost/accumulators/accumulators.hpp"
#include "boost/accumulators/statistics.hpp"
#include "boost/serialization/vector.hpp"
#include <boost/mpi.hpp>

using namespace Eigen;
using namespace boost::accumulators;
using namespace boost::math;


const int OWNER{ 0 };
const int VALUE_TAG{ 0 };
const int SUBMATRIX_TAG{ 1 };
const int MATRIX_COLS{ 2 };
const int DETERMINANT{ 3 };
const int J_COUNT{ 4 };



double simpleDeterminant(const MatrixXd& matrix);
double threadDetermenant(const MatrixXd& matrix, const uint16_t threadCount);
double mpiDeterminant(int n, boost::mpi::communicator &world, boost::mpi::environment &env);
void getMinor(const MatrixXd& matrix, MatrixXd& subMatrix, const uint16_t j);
void calculateAddition(uint16_t j, MatrixXd subMatrix, boost::atomic_ref<double> determinant, double point);
void validate(int argc, char** argv);
void validateMPI(int argc, char** argv);
void equalDeterminant(double actual, double except);
