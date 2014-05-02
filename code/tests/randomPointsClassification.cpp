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
    // generate random points
    // TODO: make sure clang complete C++11 works
    nor
    
    // calculate A,B,C coefficients, where Ax + By + C = 0 
    
}
