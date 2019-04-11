#include "pch.h"
#include "Image.h"

Image image = Image("lena.bmp");
//1111
void f3() {
	image.SaveRGB();
}

void f41() {
	double RG = image.Correlation(R, G);
	cout << "(R,G): " << RG << endl;
}
void f42() {
	double RB = image.Correlation(R, B);
	cout << "(R,B): " << RB << endl;
}
void f43() {
	double BG = image.Correlation(B, G);
	cout << "(B,G): " << BG << endl;
}


void f4() {
	cout << "4a" << endl;
	thread t1(f41);
	thread t2(f42);
	thread t3(f43);
	t1.join();
	t2.join();
	t3.join();
}

int main() {
	//f3();
	f4();
	system("pause");
	return 0;
}