#include "../Application/Utils.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

void print2dVec(vector<vector<int>>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        for (int j = 0; j < vec.at(i).size(); j++) {
            cout << ' ' << vec.at(i)[j];
        }
        cout << endl;
    }
}

vector<vector<int>> generateRandomMatrix(int N) {
    // Seed the random number generator with the current time
    // srand(time(nullptr));

    // Create a 2D vector of integers
    vector<vector<int>> vec;

    // Generate N x N matrix in vector
    for (int i = 0; i < N; i++) {
        vector<int> innerVec;
        for (int j = 0; j < N; j++) {
            // Generate a random number in the range [0,5]
            int num = rand() % 5;

            // Push num to the innerVec
            innerVec.push_back(num);
        }
        // Push innerVec to vec
        vec.push_back(innerVec);
    }

    return vec;
}

vector<vector<int>> minor(vector<vector<int>>& vec, int row, int col) {
    vector<vector<int>> result;
    for (int i = 0; i < vec.size(); i++) {
        if (i == row) continue;
        vector<int> temp;
        for (int j = 0; j < vec.at(i).size(); j++) {
            if (j == col) continue;
            temp.push_back(vec.at(i)[j]);
        }
        result.push_back(temp);
    }
    return result;
}

int determinant(vector<vector<int>>& vec) {
    if (vec.size() == 2) {
        return (vec.at(0)[0] * vec.at(1)[1]) - (vec.at(0)[1] * vec.at(1)[0]);
    }
    else {
        int result = 0;
        for (int i = 0; i < vec.at(0).size(); i++) {
            if (vec.at(0)[i] == 0) continue;
            vector<vector<int>> cof = minor(vec, 0, i);
            if ((i + 1) % 2 != 0) {
                result += vec.at(0)[i] * determinant(cof);
            }
            else {
                result -= vec.at(0)[i] * determinant(cof);
            }
        }
        return result;
    }
}

void map2dVecToEigen(vector <vector<int>>& mat, Eigen::MatrixXd& matrix) {

    std::vector<double> newMat(0);
    for (auto &row : mat) {
        for (auto rowElement : row) {
            newMat.push_back(rowElement);
        }
    }
    matrix.resize(mat.size(), mat.size());
    matrix = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(newMat.data(), mat.size(), mat.size());
};
int main(int argc, char** argv) {
    // get N from an argument pass in cmd
    int N = atoi(argv[1]);
    int sample_max = atoi(argv[2]);
    std::vector<double> timeState(0);
    std::vector<double> simpleTimeState(0);
    boost::timer::cpu_timer timer;
    // Initialization of parallel using MPI
    MPI_Init(NULL, NULL);

    // Get num of processor 
    int world_size; MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get rank of processor
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    for (size_t i = 1; i <= sample_max; i++)
    {
        vector <vector<int>> mat = generateRandomMatrix(N);



        // Task decomposition
        int start_pos = 0;
        int end_pos = 0;
        int task_per_process = N / world_size == 0 ? 1 : N / world_size;
        if (world_size >= N) {
            if (world_rank < N) {
                start_pos = task_per_process * world_rank; // inclusive
                end_pos = start_pos + task_per_process; // exclusive
            }
        }
        else {
            start_pos = task_per_process * world_rank; // inclusive
            end_pos = start_pos + task_per_process; // exclusive

            if (world_rank == world_size - 1) {
                if (N % world_size > 0 && N > world_size) {
                    end_pos += (N % world_size);
                }
                else if (end_pos == start_pos + 1 && N < world_size) {
                    end_pos = start_pos;
                }
            }
        }
        // cout << "Process " << world_rank << " start on " << start_pos << " end on " << end_pos;

        int curr_rank_result = 0;
        int global_result = 0;

        // start timer
        if (world_rank == 0) {
            timer.start();
        }
        // Parallel execution to find determinant of matrix
        if (mat.size() == 1) {
            curr_rank_result = mat.at(0)[0];
        }
        else if (mat.size() == 2) {
            curr_rank_result = (mat.at(0)[0] * mat.at(1)[1]) - (mat.at(0)[1] * mat.at(1)[0]);
        }
        else {
            for (int i = start_pos; i < end_pos; i++) {
                if (mat.at(0)[i] == 0) continue;
                vector<vector<int>> cof = minor(mat, 0, i);
                // Make cofactor
                if ((i + 1) % 2 != 0) {
                    curr_rank_result += mat.at(0)[i] * determinant(cof);
                }
                else {
                    curr_rank_result -= mat.at(0)[i] * determinant(cof);
                }
            }
        }

        if (mat.size() > 2) {
            MPI_Barrier(MPI_COMM_WORLD);
            // MPI_Allreduce(&curr_rank_result, &global_result, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Reduce(&curr_rank_result, &global_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        }

        // Stop timer
        if (world_rank == 0) {
            timer.stop();
            timeState.push_back(atof(timer.format(3, "%w").c_str()));
        }
        // Show result
        if (world_rank == 0) {
            Eigen::MatrixXd matrix;
            map2dVecToEigen(mat, matrix);
            int determinant = mat.size() > 2 ? global_result : curr_rank_result;
            timer.start();
            double simple = simpleDeterminant(matrix);
            simpleTimeState.push_back(atof(timer.format(3, "%w").c_str()));
            timer.stop();
            std::printf("Sample #%d:\n\tMPI determinant: %.3f, simple determinant: %.3f\n", i, static_cast<double>(determinant), simple);
            std::printf("\tMPI time: %.3f s, simple time: %.3f s\n\n", timeState[i], simpleTimeState[i]);
            
        }
    }
    if (world_rank == 0) {
        boost::accumulators::accumulator_set<double, features<tag::mean, tag::variance>> acc;
        boost::accumulators::accumulator_set<double, features<tag::mean, tag::variance>> accSimple;
        for (auto& time : timeState) {
            acc(time);
        }

        for (auto& time : simpleTimeState) {
            accSimple(time);
        }
        double mean{ boost::accumulators::mean(acc) };
        double simpleMean{ boost::accumulators::mean(accSimple) };
        double variance{ boost::accumulators::variance(acc) };
        double varianceSimple{ boost::accumulators::variance(accSimple) };
        double grow_value{ 100 - mean * 100 / simpleMean };
        std::printf("\nMean time MPI:%.3f\n", mean);
        std::printf("Mean time simple:%.3f\n", simpleMean);
        std::printf("Mean grow:%.3f%%\n\n", grow_value);

        students_t dist(sample_max - 1);
        double T = boost::math::quantile(complement(dist, 0.05 / 2));
        double wSimple = T * simpleMean / std::sqrt(double(simpleTimeState.size())) / 2;
        double wMPI = T * mean / std::sqrt(double(simpleTimeState.size())) / 2;


        double lowerLimitSimple = simpleMean - wSimple;
        double upperLimitSimple = simpleMean + wSimple;

        double lowerLimitMPI = mean - wMPI;
        double upperLimitMPI = mean + wMPI;


        std::printf("\t\t\t\tAll statistic:\n\n\n\tsimple_mean\tmpi_mean\n\n");
        std::printf("\t%.3f\t\t%.3f\n", simpleMean, mean);

        std::printf("Statistic for %d samples, p:%.2f%%:\n", sample_max, 95.);
        std::printf("\MPI: [mean: %.3e, variance: %.3e,confidence interval: (%.3e;%.3e)]\n", mean,
            variance, lowerLimitMPI, upperLimitMPI);
        std::printf("\tSimple: [mean: %.3e, variance: %.3e,confidence interval: (%.3e;%.3e)]\n", simpleMean,
            varianceSimple,
            lowerLimitSimple, upperLimitSimple);

    }
    MPI_Finalize();
    return 0;
}