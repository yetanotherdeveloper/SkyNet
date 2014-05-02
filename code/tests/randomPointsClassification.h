/*! Object represting an instance of random point classification class
 *  that aims to deliver input, output data , verification and other 
 *  tools to condact classification test of random data
 *
 !*/
class randomPointsClassification
{
    private:
        float minX,maxX;
        float minY,maxY;

    public:
        randomPointsClassification(unsigned int N);
    private:
        void makeRandomFunction();
};
