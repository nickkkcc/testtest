#include "Utils.h"


int main(int argc, char** argv)
{
   
    // ---------------------TESTS----------------------------
    MatrixXd a(3, 3);
    a << 1, 0, 1,
        0, 1, 0,
        3, 0, 2;
    equalDeterminant(simpleDeterminant(a), a.determinant());
    equalDeterminant(threadDetermenant(a, 6), simpleDeterminant(a));

    // ---------------------TESTS----------------------------

    validate(argc, argv);
    int n{ atoi(argv[2]) };
    int thread_max{ atoi(argv[4]) };
    int sample_max{ atoi(argv[6]) };

    boost::timer::cpu_timer timer;


    std::vector<double> simpleAlgTime(0);
    std::vector<double> simpleAlgTimeForThread(0);
    std::vector<double> simpleAlgVarianceInForThread(0);
    std::vector<double> grow(0);

    
    std::vector<double> threadAlgTime(0);
    std::vector<double> threadAlgTimeInForThread(0);
    std::vector<double> threadAlgVarianceInForThread(0);

    //Execute thread and non-thread alghorims for 1 process.
    for (size_t thread_s = 1; thread_s <= thread_max; thread_s++){

        std::printf("Thread number:[%zd]:\n\n", thread_s);
        threadAlgTime.clear();
        for (size_t sampl = 0; sampl < sample_max; sampl++){
            MatrixXd matrix{ 10 * MatrixXd::Random(n,n) };
            timer.start();
            double resultSimple{ simpleDeterminant(matrix) };
            timer.stop();

            simpleAlgTime.push_back(atof(timer.format(10, "%w").c_str()));
            std::printf("\n\tSimpleTime: %.3f s\n", simpleAlgTime.at(sampl));

            timer.start();
            double resultThread{ threadDetermenant(matrix, thread_s) };
            timer.stop();

            threadAlgTime.push_back(atof(timer.format(10, "%w").c_str()));
            std::printf("\n\tthreadTime: %.3f s\n", threadAlgTime.at(sampl));
        }


        accumulator_set<double, features<tag::mean, tag::variance>> accThreadMean;
        accumulator_set<double, features<tag::mean, tag::variance>> accSimpleMean;
        accumulator_set<double, features<tag::mean>> accMPIMean;

        for (auto var : threadAlgTime) {
             accThreadMean(var);
        }

        for (auto var : simpleAlgTime) {
            accSimpleMean(var);
        }

        threadAlgTimeInForThread.push_back(boost::accumulators::mean(accThreadMean));
        threadAlgVarianceInForThread.push_back(boost::accumulators::variance(accThreadMean));

        simpleAlgTimeForThread.push_back(boost::accumulators::mean(accSimpleMean));
        simpleAlgVarianceInForThread.push_back(boost::accumulators::variance(accSimpleMean));

        double grow_value{ 100 - boost::accumulators::mean(accThreadMean) * 100 /
                    boost::accumulators::mean(accSimpleMean) };
        grow.push_back(grow_value);


        std::printf("\nMean time thread:%.3f\n", boost::accumulators::mean(accThreadMean));
        std::printf("Mean time simple:%.3f\n", boost::accumulators::mean(accSimpleMean));
        std::printf("Mean grow:%.3f%%\n\n", grow_value);
    }

    students_t dist(thread_max - 1);
    double T = boost::math::quantile(complement(dist, 0.05 / 2));
    double wSimple = T * simpleAlgTimeForThread.at(thread_max - 1) / std::sqrt(double(simpleAlgTimeForThread.size())) / 2;
    double wThread = T * threadAlgVarianceInForThread.at(thread_max - 1) / std::sqrt(double(threadAlgTimeInForThread.size())) / 2;


    double lowerLimitSimple = simpleAlgTimeForThread.at(thread_max - 1) - wSimple;
    double upperLimitSimple = simpleAlgTimeForThread.at(thread_max - 1) + wSimple;

    double lowerLimitThread = threadAlgVarianceInForThread.at(thread_max - 1) - wThread;
    double upperLimitThread = threadAlgVarianceInForThread.at(thread_max - 1) + wThread;


    std::printf("\t\t\t\tAll statistic:\n\n\nthread_number\tthread_mean\tsimple_mean\tmpi_mean\n\n");
    for (int i = 1; i <= thread_max; i++)
    {
        std::printf("\t[%d]\t\t%.3e\t\t%.3f\t%.3e\n", i, threadAlgTimeInForThread.at(i -1 ), simpleAlgTimeForThread.at(i - 1), 0.0);
    }
    std::printf("Statistic for %d samples, p:%.2f%%:\n", sample_max, 95.);
    std::printf("\tThread: [mean: %.3e, variance: %.3e,confidence interval: (%.3e;%.3e)]\n",threadAlgTimeInForThread.at(thread_max -1),
        threadAlgVarianceInForThread.at(thread_max - 1), lowerLimitThread, upperLimitThread);
    std::printf("\tSimple: [mean: %.3e, variance: %.3e,confidence interval: (%.3e;%.3e)]\n", simpleAlgTimeForThread.at(thread_max - 1),
        simpleAlgVarianceInForThread.at(thread_max - 1),
        lowerLimitSimple, upperLimitSimple);

    return 0;
}

