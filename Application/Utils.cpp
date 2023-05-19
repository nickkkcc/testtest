#include "Utils.h"

void calculateAddition(uint16_t j, MatrixXd subMatrix, boost::atomic_ref<double> determinant, double point) {
	determinant += std::powf(-1, 1 + (j + 1)) * point* simpleDeterminant(subMatrix);
}

void getMinor(const MatrixXd& matrix, MatrixXd& subMatrix, uint16_t j)
{
	if (j == 0) {
		subMatrix = matrix.block(1, 1, matrix.rows() - 1, matrix.cols() - 1);
	}
	else if (j == matrix.cols() - 1) {

		subMatrix = matrix.block(1, 0, matrix.rows() - 1, matrix.cols() - 1);
	}
	else {
		subMatrix << matrix.block(1, 0, matrix.rows() - 1, j),
			matrix.block(1, j + 1, matrix.rows() - 1, matrix.cols() - j - 1);
	}
}
void validate(int argc, char** argv) {
	if (strcmp("-matrix_size", argv[1]) < 0 || strcmp("-max_thread_count", argv[3]) < 0 || atoi(argv[2]) == 0 || atoi(argv[4]) == 0 ||
		strcmp("-samp_size", argv[5]) < 0 || atoi(argv[6]) == 0) {
		std::cout << "Wrong arguments!\n Write mpiexec -n [numberProc] program_name.exe -matrix_size [size]{matrix: size X size} -max_thread_count [n] -samp_size [size]";
		exit(-1);
	}
}
void validateMPI(int argc, char** argv) {
	if (strcmp("-matrix_size", argv[1]) < 0 || atoi(argv[2]) == 0 ||
		strcmp("-samp_size", argv[3]) < 0 || atoi(argv[4]) == 0)  {
		std::cout << "Wrong arguments!\n Write mpiexec -n [numberProc] program_name.exe -matrix_size [size]{matrix: size X size} -samp_size [size]";
		exit(-1);
	}
}

void equalDeterminant(double actual, double excepted)
{
	if (actual != excepted) {
		std::printf("equal determinant test failed");
		exit(-1);
	}
}






