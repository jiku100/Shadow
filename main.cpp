#include "drawing.h"
#include "entropy.h"
#include "invariant.h"
#include "lab.h"
#include <stack>

#define ENTROPY 0
#define LAB 1
#define DETECTION 0


Mat src;

Point pointArray[8] =
{
	Point(1, -1),
	Point(1, 0),
	Point(0, -1),
	Point(-1, -1),
	Point(-1, 0),
	Point(-1, 1),
	Point(0, 1),
	Point(1, 1)
};

stack<Point> seed;
Mat dst;

void mouse_click(int event, int x, int y, int flags, void* pt) {
	static int i = 0;
	if (i == 0) {
		switch (event) {
		case EVENT_LBUTTONDOWN:
			Point pt = Point(x, y);
			seed.push(pt);
			
			while (!seed.empty())
			{
				Point  topPoint = seed.top();
				seed.pop();
				Vec3b& seedColor = src.at<Vec3b>(topPoint.y, topPoint.x);
				for (int i = 0; i < 8; i++)
				{

					Point p = topPoint + pointArray[i];//p is the point to grow
					int x = p.x;
					int y = p.y;

					if (x<0 || y<0 || x>src.cols - 1 || y>src.rows - 1 || dst.at<uchar>(y,x) == 255)
					{
						continue;
					}
					Vec3b& srcColor = src.at<Vec3b>(y, x);
					double srcColor_length = sqrtf((srcColor[0] * srcColor[0] + srcColor[1] * srcColor[1] + srcColor[2] * srcColor[2]));
					double seedColor_length = sqrtf((seedColor[0] * seedColor[0] + seedColor[1] * seedColor[1] + seedColor[2] * seedColor[2]));
					double distance = 1 - (srcColor[0] * seedColor[0] + srcColor[1] * seedColor[1] + srcColor[2] * seedColor[2]) / (srcColor_length * seedColor_length);
					//The second point is not growing and the absolute value of the gray difference between the point to be grown and the first seed point is less than 10, then grow
					if (distance < 3e-5)
					{
						dst.at<uchar>(y, x) = 255;//Use gray to indicate growth
					   //Put this point on the stack
						seed.push(p);
					}
				}
			}
			imshow("dst", dst);

			break;
		}
	}
}

void getSeed(Mat& src) {
	dst = Mat::zeros(src.size(), CV_8UC1);
	namedWindow("src");
	imshow("src", src);
	setMouseCallback("src", mouse_click);	
}

void makeSeed(Mat& src, Mat& dst) {	
	while (!seed.empty())
	{
		Point  topPoint = seed.top();
		seed.pop();
		Vec3b& seedColor = src.at<Vec3b>(topPoint.y, topPoint.x);
		for (int i = 0; i < 8; i++)
		{

			Point p = topPoint + pointArray[i];//p is the point to grow
			int x = p.x;
			int y = p.y;

			if (x<0 || y<0 || x>src.cols - 1 || y>src.rows - 1)
			{

				continue;
			}
			Vec3b& srcColor = src.at<Vec3b>(y, x);
			double srcColor_length = powf((srcColor[0] ^ 2 + srcColor[1] ^ 2 + srcColor[2] ^ 2), 1. / 3);
			double seedColor_length = powf((seedColor[0] ^ 2 + seedColor[1] ^ 2 + seedColor[2] ^ 2), 1. / 3);
			double distance = 1 - srcColor.dot(seedColor) / (srcColor_length * seedColor_length);
			//The second point is not growing and the absolute value of the gray difference between the point to be grown and the first seed point is less than 10, then grow
			if (distance < 0.1)
			{
				dst.at<uchar>(y, x) = 255;//Use gray to indicate growth
			   //Put this point on the stack
				seed.push(p);
			}
		}
	}
	imshow("dst", dst);
}

int main(void) {
#if ENTROPY == 1
	Mat src = imread("./src/shadow2.jpg");
	Size src_size = src.size();
	resize(src, src, Size(320, 240), 0, 0, INTER_CUBIC);
	imshow("src", src);
	vector<Point2f> lcs;
	vector<double> entropy;
	int angle;

	RGB2LCS(src, lcs);
	Mat LCS_M;
	drawLCS(lcs, LCS_M);
	getEntropy(lcs, entropy);
	Mat dst;
	drawEntropy(entropy, angle, dst);
	vector<int> gray_T;
	get_invariant(lcs, angle, gray_T);
	Mat invariant;
	drawInvariant(src, invariant, gray_T);
	resize(invariant, invariant, src_size, 0, 0, INTER_CUBIC);
	Mat edge;
	Canny(invariant, edge, 5, 15);
	//morphologyEx(edge, edge, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
	imshow("edge", edge);

	Mat src_Gray;
	resize(src, src, src_size, 0, 0, INTER_CUBIC);
	cvtColor(src, src_Gray, COLOR_BGR2GRAY);
	GaussianBlur(src_Gray, src_Gray, Size(), 1);
	Mat src_edge;
	Canny(src_Gray, src_edge, 20, 60);
	//morphologyEx(src_edge, src_edge, MORPH_CLOSE, Mat(), Point(-1,-1), 2);
	imshow("src_edge", src_edge);

	Mat shadow;
	bitwise_xor(edge, src_edge, shadow);
	imshow("shadow", shadow);
	waitKey();
	destroyAllWindows();

#endif
#if LAB == 1
	Mat src = imread("./src/shadow4.jpg");
	resize(src, src, Size(640, 480), 0, 0, INTER_CUBIC);
	imshow("src", src);
	Mat lab;
	cvtColor(src, lab, COLOR_BGR2Lab);		
	// -> L 0 ~ 255, A ->  1 ~ 255, B -> 1 ~ 255 
	vector<Mat> lab_planes;
	split(lab, lab_planes);
	Mat dst;
	LAB_Shadow(lab_planes, dst);

	waitKey();
	destroyAllWindows();
#endif
#if DETECTION == 1
	src = imread("./src/shadow.jpg");
	resize(src, src, Size(), 0.5, 0.5, INTER_AREA);
	GaussianBlur(src, src, Size(), 3.0);
	getSeed(src);
	//makeSeed(src, dst);
	waitKey();
	destroyAllWindows();
#endif

}
