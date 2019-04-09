#include "pch.h"
template<typename T> using matrix = vector<vector<T>>;
/// <summary>
/// 
/// </summary>
enum Color {
	R, G, B
};
/// <summary>
/// 
/// </summary>
class Pixel {
private:
	void Init(double R, double G, double B) {
		this->R = R;
		this->B = B;
		this->G = G;
	}
	void Init(size_t R, size_t G, size_t B) {
		this->R = R;
		this->B = B;
		this->G = G;
	}
	void Init(BYTE R, BYTE G, BYTE B) {
		this->R = R;
		this->B = B;
		this->G = G;
	}
public:
#pragma region FIELDS
	BYTE R;
	BYTE G;
	BYTE B;
#pragma endregion
	Pixel() {
		this->Init(0.0, 0.0, 0.0);
	}
	Pixel(double R, double G, double B) {
		this->Init(R, G, B);
	}
	Pixel(size_t R, size_t G, size_t B) {
		this->Init(R, G, B);
	}
	Pixel(BYTE R, BYTE G, BYTE B) {
		this->Init(R, G, B);
	}
	BYTE Get(Color color) {
		if (color == Color::R)
			return this->R;
		if (color == Color::G)
			return this->G;
		if (color == Color::B)
			return this->B;
		return
			0X00;
	}
};
template <typename T>
/// <summary>
/// 
/// </summary>
class Matrix
{
private:
public:
	vector<vector<T>> items;
	Matrix() {

	}
	Matrix(size_t height, size_t width) {
		for (int i = 0; i < height; i++)
		{
			vector<T> line;
			for (int j = 0; j < width; j++)
				line.push_back(T());
			items.push_back(line);
		}
	}
	size_t GetRowsCount() { return this->items.size(); }
	size_t GetColumnsCount() { if (this->GetColumnsCount() > 0)return this->items[0].size(); }
};

/// <summary>
/// 
/// </summary>
class Image {
private:
	string RGB = "RGB";
	void Init(string filename) {
		FILE* image;
		fopen_s(&image, filename.c_str(), "rb");
		fread(&this->fileHeader, sizeof(this->fileHeader), 1, image);
		fread(&this->infoHeader, sizeof(this->infoHeader), 1, image);
		this->height = this->infoHeader.biHeight;
		this->width = this->infoHeader.biWidth;
		if ((this->width * 3) % 4)
			this->padding = 4 - (this->width * 3) % 4;
		for (int i = 0; i < this->height; i++) {
			RGBTRIPLE *line = new RGBTRIPLE[this->width];
			fread(&line[0], sizeof(RGBTRIPLE), this->width, image);
			if (this->padding != 0)
				fread(&line[0], this->padding, 1, image);
			vector<Pixel>pixels;
			for (int j = 0; j < this->width; j++)
				pixels.push_back(Pixel(line[j].rgbtRed, line[j].rgbtGreen, line[j].rgbtBlue));
			this->pixels.items.push_back(pixels);
		}
		this->padding = padding;
		fclose(image);
	}
	void SaveRGB(Color color) {
		FILE *image;
		stringstream ss;
		ss << "out" << RGB[(int)color] << ".bmp";
		string filename = ss.str();
		fopen_s(&image, filename.c_str(), "wb");
		fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, image);
		fwrite(&infoHeader, sizeof(BITMAPINFOHEADER), 1, image);
		for (int i = 0; i < height; i++) {
			RGBTRIPLE pixel;
			for (int j = 0; j < width; j++) {
				if (color == Color::R)
					pixel.rgbtRed = this->pixels.items[i][j].R;
				else
					pixel.rgbtRed = 0X00;
				if (color == Color::G)
					pixel.rgbtGreen = this->pixels.items[i][j].G;
				else
					pixel.rgbtGreen = 0X00;
				if (color == Color::B)
					pixel.rgbtBlue = this->pixels.items[i][j].B;
				else
					pixel.rgbtBlue = 0X00;
				fwrite(&pixel, sizeof(RGBTRIPLE), 1, image);
				if (i == 0 && padding)
					fwrite(&pixel, padding, 1, image);
			}
		}
		fclose(image);
	}
	double Expected(Matrix<double> values) {
		double result = 0;
		for (int i = 0; i < this->height; i++)
			for (int j = 0; j < this->width; j++)
				result += values.items[i][j];
		result /= (this->height*this->width);
		return result;
	}
	double Expected(Color color) {
		double result = 0;
		for (int i = 0; i < this->height; i++)
			for (int j = 0; j < this->width; j++)
				result += this->pixels.items[i][j].Get(color);
		result /= (this->height*this->width);
		return result;
	}
	double Sigma(Color color) {
		double expected = this->Expected(color);
		double sigma = 0;
		for (int i = 0; i < this->height; i++)
			for (int j = 0; j < this->width; j++)
				sigma += (this->pixels.items[i][j].Get(color) - expected) *
				(this->pixels.items[i][j].Get(color) - expected);
		return sqrt(sigma / (this->height * this->width - 1));
	}
public:
	BITMAPFILEHEADER fileHeader;
	BITMAPINFOHEADER infoHeader;
	size_t height;
	size_t width;
	size_t padding;
	Matrix<Pixel> pixels;
	Image() {
		fileHeader = { 0 };
		infoHeader = { 0 };
		padding = 0;
	}
	Image(string filename) {
		Init(filename);
	}
	void SaveRGB() {
		SaveRGB(Color::R);
		SaveRGB(Color::G);
		SaveRGB(Color::B);
	}
	double Correlation(Color color1, Color color2) {
		double expectedA = this->Expected(color1);
		double expectedB = this->Expected(color2);
		Matrix<double> values(this->height, this->width);
		for (int i = 0; i < this->height; i++)
			for (int j = 0; j < this->width; j++)
				values.items[i][j] = ((this->pixels.items[i][j].Get(color1) - expectedA)
					* (this->pixels.items[i][j].Get(color2) - expectedB));
		double expected = this->Expected(values) / (this->Sigma(color1) * this->Sigma(color2));
		return expected;
	}
};

void readFile(const char* filename, BITMAPFILEHEADER &file_header, BITMAPINFOHEADER &info_header, size_t &padding, matrix<double> &R, matrix<double> &G, matrix<double> &B) {
	FILE* image;
	fopen_s(&image, filename, "rb");
	fread(&file_header, sizeof(file_header), 1, image);
	fread(&info_header, sizeof(info_header), 1, image);
	if ((info_header.biWidth * 3) % 4) padding = 4 - (info_header.biWidth * 3) % 4;
	RGBTRIPLE **triple = new RGBTRIPLE *[info_header.biHeight];
	for (int i = 0; i < info_header.biHeight; i++) {
		triple[i] = new RGBTRIPLE[info_header.biWidth];
		fread(&triple[i][0], sizeof(RGBTRIPLE), info_header.biWidth, image);
		if (padding != 0) {
			fread(&triple[i][0], padding, 1, image);
		}
	}
	fclose(image);

	for (int i = 0; i < info_header.biHeight; i++) {
		vector<double> Rvector;
		vector<double> Gvector;
		vector<double> Bvector;
		for (int j = 0; j < info_header.biWidth; j++) {
			Rvector.push_back(static_cast<double>(triple[i][j].rgbtRed));
			Gvector.push_back(static_cast<double>(triple[i][j].rgbtGreen));
			Bvector.push_back(static_cast<double>(triple[i][j].rgbtBlue));
		}
		R.push_back(Rvector);
		G.push_back(Gvector);
		B.push_back(Bvector);
	}

	for (int i = 0; i < info_header.biHeight; i++) {
		delete[] triple[i];
	}
	delete[] triple;
}

void fileRGB(BITMAPFILEHEADER file_header, BITMAPINFOHEADER info_header, size_t padding, matrix<double> R, matrix<double> G, matrix<double> B) {
	FILE *R_file, *G_file, *B_file;
	fopen_s(&R_file, "outR.bmp", "wb");
	fopen_s(&G_file, "outG.bmp", "wb");
	fopen_s(&B_file, "outB.bmp", "wb");
	fwrite(&file_header, sizeof(BITMAPFILEHEADER), 1, R_file);
	fwrite(&info_header, sizeof(BITMAPINFOHEADER), 1, R_file);
	fwrite(&file_header, sizeof(BITMAPFILEHEADER), 1, G_file);
	fwrite(&info_header, sizeof(BITMAPINFOHEADER), 1, G_file);
	fwrite(&file_header, sizeof(BITMAPFILEHEADER), 1, B_file);
	fwrite(&info_header, sizeof(BITMAPINFOHEADER), 1, B_file);

	RGBTRIPLE **R_comp = new RGBTRIPLE *[info_header.biHeight];
	RGBTRIPLE **G_comp = new RGBTRIPLE *[info_header.biHeight];
	RGBTRIPLE **B_comp = new RGBTRIPLE *[info_header.biHeight];
	for (int i = 0; i < info_header.biHeight; i++) {
		R_comp[i] = new RGBTRIPLE[info_header.biWidth];
		G_comp[i] = new RGBTRIPLE[info_header.biWidth];
		B_comp[i] = new RGBTRIPLE[info_header.biWidth];
		for (int j = 0; j < info_header.biWidth; j++) {
			R_comp[i][j].rgbtRed = static_cast<BYTE>(R[i][j]);
			R_comp[i][j].rgbtGreen = 0x00;
			R_comp[i][j].rgbtBlue = 0x00;
			fwrite(&R_comp[i][j], sizeof(RGBTRIPLE), 1, R_file);

			G_comp[i][j].rgbtRed = 0x00;
			G_comp[i][j].rgbtGreen = static_cast<BYTE>(G[i][j]);
			G_comp[i][j].rgbtBlue = 0x00;
			fwrite(&G_comp[i][j], sizeof(RGBTRIPLE), 1, G_file);

			B_comp[i][j].rgbtRed = 0x00;
			B_comp[i][j].rgbtGreen = 0x00;
			B_comp[i][j].rgbtBlue = static_cast<BYTE>(B[i][j]);
			fwrite(&B_comp[i][j], sizeof(RGBTRIPLE), 1, B_file);
		}
		if (padding != 0) {
			fwrite(&R_comp[i][0], padding, 1, R_file);
			fwrite(&G_comp[i][0], padding, 1, G_file);
			fwrite(&B_comp[i][0], padding, 1, B_file);
		}
	}
	fclose(R_file);
	fclose(G_file);
	fclose(B_file);

	for (int i = 0; i < info_header.biHeight; i++) {
		delete[] R_comp[i];
		delete[] G_comp[i];
		delete[] B_comp[i];
	}
	delete[] R_comp;
	delete[] G_comp;
	delete[] B_comp;
}

void fileYCbCr(BITMAPFILEHEADER file_header, BITMAPINFOHEADER info_header, size_t padding, matrix<double> Y, matrix<double> Cb, matrix<double> Cr) {
	FILE *Y_file, *Cb_file, *Cr_file;
	fopen_s(&Y_file, "outY.bmp", "wb");
	fopen_s(&Cb_file, "outCb.bmp", "wb");
	fopen_s(&Cr_file, "outCr.bmp", "wb");
	fwrite(&file_header, sizeof(BITMAPFILEHEADER), 1, Y_file);
	fwrite(&info_header, sizeof(BITMAPINFOHEADER), 1, Y_file);
	fwrite(&file_header, sizeof(BITMAPFILEHEADER), 1, Cb_file);
	fwrite(&info_header, sizeof(BITMAPINFOHEADER), 1, Cb_file);
	fwrite(&file_header, sizeof(BITMAPFILEHEADER), 1, Cr_file);
	fwrite(&info_header, sizeof(BITMAPINFOHEADER), 1, Cr_file);

	RGBTRIPLE **Y_comp = new RGBTRIPLE *[info_header.biHeight];
	RGBTRIPLE **Cb_comp = new RGBTRIPLE *[info_header.biHeight];
	RGBTRIPLE **Cr_comp = new RGBTRIPLE *[info_header.biHeight];
	for (int i = 0; i < info_header.biHeight; i++) {
		Y_comp[i] = new RGBTRIPLE[info_header.biWidth];
		Cb_comp[i] = new RGBTRIPLE[info_header.biWidth];
		Cr_comp[i] = new RGBTRIPLE[info_header.biWidth];
		for (int j = 0; j < info_header.biWidth; j++) {
			Y_comp[i][j].rgbtRed = static_cast<BYTE>(Y[i][j]);
			Y_comp[i][j].rgbtGreen = static_cast<BYTE>(Y[i][j]);
			Y_comp[i][j].rgbtBlue = static_cast<BYTE>(Y[i][j]);
			fwrite(&Y_comp[i][j], sizeof(RGBTRIPLE), 1, Y_file);

			Cb_comp[i][j].rgbtRed = static_cast<BYTE>(Cb[i][j]);
			Cb_comp[i][j].rgbtGreen = static_cast<BYTE>(Cb[i][j]);
			Cb_comp[i][j].rgbtBlue = static_cast<BYTE>(Cb[i][j]);
			fwrite(&Cb_comp[i][j], sizeof(RGBTRIPLE), 1, Cb_file);

			Cr_comp[i][j].rgbtRed = static_cast<BYTE>(Cr[i][j]);
			Cr_comp[i][j].rgbtGreen = static_cast<BYTE>(Cr[i][j]);
			Cr_comp[i][j].rgbtBlue = static_cast<BYTE>(Cr[i][j]);
			fwrite(&Cr_comp[i][j], sizeof(RGBTRIPLE), 1, Cr_file);
		}
		if (padding != 0) {
			fwrite(&Y_comp[i][0], padding, 1, Y_file);
			fwrite(&Cb_comp[i][0], padding, 1, Cb_file);
			fwrite(&Cr_comp[i][0], padding, 1, Cr_file);
		}
	}
	fclose(Y_file);
	fclose(Cb_file);
	fclose(Cr_file);

	for (int i = 0; i < info_header.biHeight; i++) {
		delete[] Y_comp[i];
		delete[] Cb_comp[i];
		delete[] Cr_comp[i];
	}
	delete[] Y_comp;
	delete[] Cb_comp;
	delete[] Cr_comp;
}

void fullCompFile(matrix<double> R, matrix<double> G, matrix<double> B) {

	DWORD padding = ((4 - (R[0].size() * 3) % 4) & 3);
	DWORD size = 3 * static_cast<DWORD>(R.size()) * static_cast<DWORD>(R[0].size()) + static_cast<DWORD>(padding) * static_cast<DWORD>(R.size());

	BITMAPFILEHEADER file_header = { 19778, size + 54, 0, 0, 54 };
	BITMAPINFOHEADER info_header = { 40, static_cast<LONG>(R[0].size()), static_cast<LONG>(R.size()), 1, 24, 0, size, 0, 0, 0, 0 };

	FILE *file;
	fopen_s(&file, "rotate.bmp", "wb");
	fwrite(&file_header, sizeof(BITMAPFILEHEADER), 1, file);
	fwrite(&info_header, sizeof(BITMAPINFOHEADER), 1, file);

	RGBTRIPLE **triple = new RGBTRIPLE *[info_header.biHeight];
	for (int i = 0; i < info_header.biHeight; i++) {
		triple[i] = new RGBTRIPLE[info_header.biWidth];
		for (int j = 0; j < info_header.biWidth; j++) {
			triple[i][j].rgbtRed = static_cast<BYTE>(R[i][j]);
			triple[i][j].rgbtGreen = static_cast<BYTE>(G[i][j]);
			triple[i][j].rgbtBlue = static_cast<BYTE>(B[i][j]);

			fwrite(&triple[i][j], sizeof(RGBTRIPLE), 1, file);
		}
		if (padding != 0) {
			fwrite(&triple[i][0], padding, 1, file);
		}
	}
	fclose(file);

	for (int i = 0; i < info_header.biHeight; i++) {
		delete[] triple[i];
	}
	delete[] triple;
}

double expected(matrix<double> component, int H, int W) {
	double expected_value = 0;
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			expected_value += component[i][j];
		}
	}
	expected_value /= (H * W);
	return expected_value;
}

double sigma(matrix<double> component, int H, int W) {
	double component_expected = expected(component, H, W);

	double sigma_value = 0;
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			sigma_value += (component[i][j] - component_expected) * (component[i][j] - component_expected);
		}
	}

	return sqrt(sigma_value / (H * W - 1));
}

double correlation(matrix<double> A, matrix<double> B, int H, int W) {
	if (H == 0) {
		H = static_cast<int>(A.size());
		W = static_cast<int>(A[0].size());
	}
	double Aexpected = expected(A, H, W);
	double Bexpected = expected(B, H, W);

	matrix<double> common_array;
	for (int i = 0; i < H; i++) {
		vector<double> tmp_vector;
		for (int j = 0; j < W; j++) {
			tmp_vector.push_back((A[i][j] - Aexpected) * (B[i][j] - Bexpected));
		}
		common_array.push_back(tmp_vector);
	}

	double expected_value = expected(common_array, H, W);
	double sigmaA = sigma(A, H, W);
	double sigmaB = sigma(B, H, W);
	expected_value /= (sigmaA * sigmaB);
	return expected_value;
}

double clipping(double value) {
	if (value > 255) return 255;
	else if (value < 0) return 0;
	return value;
}

void RGBToYCbCr(matrix<double> R, matrix<double> G, matrix<double> B, matrix<double> &Y, matrix<double> &Cb, matrix<double> &Cr) {
	for (int i = 0; i < static_cast<int>(R.size()); i++) {
		vector<double> Yvec;
		vector<double> Cbvec;
		vector<double> Crvec;
		for (int j = 0; j < static_cast<int>(R[0].size()); j++) {
			double tmpY = round(clipping((0.299 * R[i][j]) + (0.587 * G[i][j]) + (0.114 * B[i][j])));
			Yvec.push_back(tmpY);
			Cbvec.push_back(round(clipping((0.5643 * (B[i][j] - tmpY)) + 128.0)));
			Crvec.push_back(round(clipping((0.7132 * (R[i][j] - tmpY)) + 128.0)));
		}
		Y.push_back(Yvec);
		Cb.push_back(Cbvec);
		Cr.push_back(Crvec);
	}
}

void YCbCrToRGB(matrix<double> Y, matrix<double> Cb, matrix<double> Cr, matrix<double> &R, matrix<double> &G, matrix<double> &B) {
	for (int i = 0; i < static_cast<int>(Y.size()); i++) {
		vector<double> Rvec;
		vector<double> Gvec;
		vector<double> Bvec;
		for (int j = 0; j < static_cast<int>(Y[0].size()); j++) {
			Rvec.push_back(round(clipping(Y[i][j] + (1.402 * (Cr[i][j] - 128.0)))));
			Gvec.push_back(round(clipping(Y[i][j] - (0.714 * (Cr[i][j] - 128.0)) - (0.334 * (Cb[i][j] - 128.0)))));
			Bvec.push_back(round(clipping(Y[i][j] + (1.772 * (Cb[i][j] - 128.0)))));
		}
		R.push_back(Rvec);
		G.push_back(Gvec);
		B.push_back(Bvec);
	}
}

double PSNR(matrix<double> original, matrix<double> recovery) {
	double numerator = static_cast<double>(original.size()) * static_cast<double>(original[0].size());
	numerator *= 65025.0;
	double denominator = 0;

	for (int i = 0; i < static_cast<int>(original.size()); i++) {
		for (int j = 0; j < static_cast<int>(original[0].size()); j++) {
			denominator += ((original[i][j] - recovery[i][j]) * (original[i][j] - recovery[i][j]));
		}
	}

	return 10.0 * log10(numerator / denominator);
}

matrix<double> decimation8a(matrix<double> C) {
	matrix<double> result;
	for (int i = 1; i < static_cast<int>(C.size()); i += 2) {
		vector<double> CI;
		for (int j = 1; j < static_cast<int>(C[0].size()); j += 2) {
			CI.push_back(C[i][j]);
		}
		result.push_back(CI);
	}
	return result;
}

matrix<double> decimation8b(matrix<double> C) {
	matrix<double> result;
	for (int i = 0; i < static_cast<int>(C.size()); i += 2) {
		vector<double> CI;
		for (int j = 0; j < static_cast<int>(C[0].size()); j += 2) {
			if (i + 1 < static_cast<int>(C.size())) {
				CI.push_back((C[i][j] + C[i + 1][j] + C[i][j + 1] + C[i + 1][j + 1]) / 4);
			}
			else {
				CI.push_back((C[i][j] + C[i][j + 1]) / 2);
				CI.push_back((C[i][j] + C[i][j + 1]) / 2);
			}
		}
		result.push_back(CI);
	}
	return result;
}

matrix<double> recovering(matrix<double> C) {
	matrix<double> result;
	for (int i = 0; i < static_cast<int>(C.size()); i++) {
		vector<double> CIFirst;
		vector<double> CISecond;
		for (int j = 0; j < static_cast<int>(C[0].size()); j++) {
			CIFirst.push_back(C[i][j]);
			CISecond.push_back(C[i][j]);

			if (i == 0 && j == 0) {
				CIFirst.push_back(C[i][j]);
				CISecond.push_back(C[i][j]);
			}
			else if (i == 0) {
				CIFirst.push_back(C[i][j - 1]);
				CISecond.push_back((C[i][j - 1] + CISecond.back()) / 2);
			}
			else if (j == 0) {
				CIFirst.push_back(C[i - 1][j]);
				CISecond.push_back(C[i - 1][j]);
			}
			else {
				CIFirst.push_back((C[i][j - 1] + C[i - 1][j]) / 2);
				CISecond.push_back((CIFirst.back() + CISecond.back()) / 2);
			}

		}
		result.push_back(CIFirst);
		result.push_back(CISecond);
	}
	return result;
}

void frequencyHistogram(const char* filename, matrix<double> component) {

	ofstream file(filename, ios_base::out | ios_base::binary);

	for (int x = 0; x < 256; x++) {
		int count = 0;
		for (int i = 0; i < static_cast<int>(component.size()); i++) {
			for (int j = 0; j < static_cast<int>(component[0].size()); j++) {
				if (component[i][j] == x) {
					count++;
				}
			}
		}
		file << count << " ";
	}
	file.close();
}

double entropy(matrix<double> component) {
	double result = 0;
	for (int x = 0; x < 256; x++) {
		int count = 0;
		for (int i = 0; i < static_cast<int>(component.size()); i++) {
			for (int j = 0; j < static_cast<int>(component[0].size()); j++) {
				if (component[i][j] == x) {
					count++;
				}
			}
		}
		double p = (count / (static_cast<double>(component.size()) * static_cast<double>(component[0].size())));
		if (p != 0) result += (p * log2(p));
	}
	return (-result);
}

matrix<double> DArray(matrix<double> component, int rule_number) {
	matrix<double> result;

	for (int i = 1; i < static_cast<double>(component.size()); i++) {
		vector<double> resI;
		for (int j = 1; j < static_cast<double>(component[0].size()); j++) {
			double f = 0;
			if (rule_number == 1) f = component[i][j - 1];
			else if (rule_number == 2) f = component[i - 1][j];
			else if (rule_number == 3) f = component[i - 1][j - 1];
			else if (rule_number == 4) f = ((component[i][j - 1] + component[i - 1][j] + component[i - 1][j - 1]) / 3);
			resI.push_back(component[i][j] - f);
		}
		result.push_back(resI);
	}
	return result;
}

void rotate(matrix<double> R, matrix<double> G, matrix<double> B, matrix<double> &rotateR, matrix<double> &rotateG, matrix<double> &rotateB) {
	for (int j = static_cast<int>(R[0].size()) - 1; j >= 0; j--) {
		vector<double> rotateRI;
		vector<double> rotateGI;
		vector<double> rotateBI;
		for (int i = 0; i < static_cast<int>(R.size()); i++) {
			rotateRI.push_back(R[i][j]);
			rotateGI.push_back(G[i][j]);
			rotateBI.push_back(B[i][j]);
		}
		rotateR.push_back(rotateRI);
		rotateG.push_back(rotateGI);
		rotateB.push_back(rotateBI);
	}
}

matrix<double> shifting(matrix<double> &component) {
	int H = static_cast<int>(component.size());
	int W = static_cast<int>(component[0].size());

	matrix<double> component_2;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < H; j++) {
			vector<double> vec;
			for (auto &k : component[j])
				vec.push_back(k);
			component_2.push_back(vec);
		}
	}

	matrix<double> corrCoef;
	for (int y = -10; y <= 10; y += 5) {
		vector<double> corrCoef_vec;
		for (int x = 0; x < W / 4; x += 4) {
			matrix<double> Aij;
			matrix<double> Amn;

			for (int i = H; i < (2 * H - y); i++) {
				vector<double> vec;
				vec.reserve(static_cast<unsigned long>(W - x));
				for (int j = 0; j < (W - x); j++)
					vec.push_back(component_2[i][j]);
				Aij.push_back(vec);
			}

			for (int m = (H + y); m < (2 * H); m++) {
				vector<double> vec;
				for (int n = x; n < W; n++)
					vec.push_back(component_2[m][n]);
				Amn.push_back(vec);
			}

			corrCoef_vec.push_back(correlation(Aij, Amn, H - y, W - x));
		}
		corrCoef.push_back(corrCoef_vec);
	}
	return corrCoef;
}

void shifting(matrix<double> R, matrix<double> G, matrix<double> B) {
	matrix<double> R_corr_coef = shifting(R);
	matrix<double> G_corr_coef = shifting(G);
	matrix<double> B_corr_coef = shifting(B);

	ofstream fsR("R_corr_coef", ios_base::out | ios_base::binary);
	ofstream fsG("G_corr_coef", ios_base::out | ios_base::binary);
	ofstream fsB("B_corr_coef", ios_base::out | ios_base::binary);

	for (const auto &vector : R_corr_coef) {
		for (const auto &cc : vector)
			fsR << cc << ' ';
		fsR << endl;
	}

	for (const auto &vector : G_corr_coef) {
		for (const auto &cc : vector)
			fsG << cc << ' ';
		fsG << endl;
	}

	for (const auto &vector : B_corr_coef) {
		for (const auto &cc : vector)
			fsB << cc << ' ';
		fsB << endl;
	}
}