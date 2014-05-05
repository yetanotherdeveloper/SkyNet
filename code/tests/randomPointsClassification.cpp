#include <random>
#include "randomPointsClassification.h"

/*!
 * \param  Generate
   !*/
randomPointsClassification::randomPointsClassification(unsigned int N)
{
    // Area from which random points are to be sampled
    this->minX = -1.0f;
    this->maxX =  1.0f;
    this->minY = -1.0f;
    this->maxY =  1.0f;

    makeRandomFunction();
}

/*! Idea is to take random two points on <min_x,rand_y> and <max_x,rand_y>
 */
void randomPointsClassification::makeRandomFunction()
{
    std::uniform_real_distribution<float> ud(0.0f,1.0f);
    std::random_device rd;

    // First random point generation <min_x,rand_y>
    float randval = ud(rd);
    float r1x     = this->minX;
    float r1y     = this->minY + (this->maxY - this->minY) * randval;

    // second random point generation <max_x,rand_y>
    randval = ud(rd);
    float r2x = this->maxX;
    float r2y = this->minY + (this->maxY - this->minY) * randval;

    // calculate A,B,C coefficients, where Ax + By + C = 0 
    // y = (y_2 - y_1)/(x_2 - x_1)*(x - x_1) <=>
    // <=>  (x_2 - x_1)*y + (y_1 -y_2)*x + y_2*x_1 - y_1*x_1 = 0
    this->A = r1y - r2y;
    this->B = r2x - r1x;
    this->C = r2y * r1x - r1y * r1x;
}
