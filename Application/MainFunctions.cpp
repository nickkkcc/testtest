#include "Utils.h"



double simpleDeterminant(const MatrixXd& matrix)
{
	if (matrix.size() == 4) {
		return matrix(0, 0) * matrix(1, 1) -
			matrix(0, 1) * matrix(1, 0);
	}

	if (matrix.size() == 1) {
		return matrix(0, 0);
	}

	double determinant = 0;
	MatrixXd subMatrix;
	for (size_t j = 0; j < matrix.cols(); j++) {
		subMatrix = MatrixXd::Zero(matrix.rows() - 1, matrix.cols() - 1);
		getMinor(matrix, subMatrix, j);
		double postDeterminant = simpleDeterminant(subMatrix);
		determinant += std::powf(-1, 1 + (j + 1)) * matrix(0, j) * postDeterminant;
	}
	return determinant;
}

double threadDetermenant(const MatrixXd& matrix, const uint16_t threadCount)
{
	int htc = boost::thread::hardware_concurrency();
	if (threadCount > htc) {
		std::printf("Превышено кол-во потоков. Выбрано %d из %d", threadCount, htc);
		exit(-1);
	}

	boost::asio::thread_pool pool(threadCount);

	double determinant = 0;

	for (size_t j = 0; j < matrix.cols(); j++)
	{
		MatrixXd subMatrix{ MatrixXd::Zero(matrix.rows() - 1, matrix.cols() - 1) };
		getMinor(matrix, subMatrix, j);

		boost::asio::defer(pool, boost::bind(calculateAddition, j, subMatrix, boost::atomic_ref<double>(determinant), matrix(0, j)));
	}
	pool.join();
	return determinant;
}

double mpiDeterminant(int n, boost::mpi::communicator& world, boost::mpi::environment& env)
{
    Eigen::MatrixXd matrix(n, n);
    int subMatrixSize = (matrix.cols() - 1) * (matrix.rows() - 1);
    Eigen::MatrixXd subMatrix(matrix.rows() - 1, matrix.cols() - 1);


    double value{ 0 };
    double determinant{ 0 };
    std::vector<double> total_determinant(world.size());
    double common_determenant{ 0 };
    const int matrixCols = matrix.cols() - 1;
    const int matrixRows = matrix.rows() - 1;

    if (world.rank() == OWNER) {
        matrix = 10 * Eigen::MatrixXd::Random(n, n);
        //std::cout << matrix << std::endl << std::endl;
    }

    for (int j = 0; j <= matrixCols;) {
        if (world.rank() == OWNER) {
            for (int worker = OWNER + 1; worker < world.size(); worker++)
            {
                value = std::powf(-1, 1 + (j + 1)) * matrix(0, j);
                getMinor(matrix, subMatrix, j);
                std::vector<double> subMatrixSTL(subMatrix.data(), subMatrix.data() + subMatrixSize);
                j++;
                world.send(worker, VALUE_TAG, &value, 1);
                world.send(worker, SUBMATRIX_TAG, &subMatrixSTL[0], subMatrixSize);
            }

                value = std::powf(-1, 1 + (j + 1)) * matrix(0, j);
                getMinor(matrix, subMatrix, j);
                j++;
                common_determenant += value * simpleDeterminant(subMatrix);
                //std::cout << "Worker #" << world.rank() << ", value: " << value << ", minor: \n" << subMatrix << std::endl << std::endl;

            for (int i = OWNER + 1; i < world.size(); i++)
            {
                world.send(i, J_COUNT, &j, 1);
            }
            for (size_t i = 0; i < world.size() - 1 && world.size() != 1; i++)
            {
                world.recv(MPI_ANY_SOURCE, DETERMINANT, &determinant, 1);
                common_determenant += determinant;
            }

        }
        else {
            std::vector<double> subMatrixSTL(subMatrixSize);
            world.recv(OWNER, VALUE_TAG, &value, 1);
            world.recv(OWNER, SUBMATRIX_TAG, &subMatrixSTL[0], subMatrixSize);
            world.recv(OWNER, J_COUNT, &j, 1);
            subMatrix = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(subMatrixSTL.data(), matrixRows, matrixCols);
            //std::cout << "Worker #" << world.rank() << ", value: " << value << ", minor: \n" << subMatrix << std::endl << std::endl;
            determinant = value * simpleDeterminant(subMatrix);
            world.send(OWNER, DETERMINANT, &determinant, 1);
        }

    }
    if (world.rank() == OWNER) {
        std::printf("Determinant is %.3f and simple %.3f\n\n", common_determenant, simpleDeterminant(matrix));
        return common_determenant;
    }
    else {
        return  0.;
    }
}