#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;



int main(int argc, char **argv) 
{
    if(argc != 3)
    {
      cout << "usage:feature_extraction img1 img2" << endl;
      return 1;
    }
    
    //read image
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    //init 
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1,descriptors_2;
    Ptr<ORB> orb = ORB::create(500,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
    
    //first step Oriented FAST
    orb->detect(img_1,keypoints_1);
    orb->detect(img_2,keypoints_2);
    
    //second step BRIEF
    orb->compute(img_1,keypoints_1,descriptors_1);
    orb->compute(img_2,keypoints_2,descriptors_2);
    
    Mat outimg1;
    drawKeypoints(img_1,keypoints_1,outimg1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    imshow("ORB features",outimg1);
    
    //3th step match images
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1,descriptors_2,matches);
    
    //4th strp match point filter
    double min_dist = 10000,max_dist = 0;
    
    for(int i =0;i<descriptors_1.rows;i++)
    {
      double dist = matches[i].distance;
      if(dist <min_dist)
	 min_dist = dist;
      if(dist > max_dist)
	 max_dist =dist;
      
      
    }
    
    printf("--Max dist: %f \n",max_dist);
    printf("--Min dist: %f \n",min_dist);
    
    //when descriptor_distance <= 2*min_distance,match wrong
    //set the experienced_value to the floor  
    std::vector<DMatch> good_matches;
    for(int i = 0; i<descriptors_1.rows;i++)
    {
      if(matches[i].distance<= max(2*min_dist,30.0))
      {
	good_matches.push_back(matches[i]);
      }
    }
    
    //5th step draw the match result
    Mat img_match;
    Mat img_good_match;
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_match);
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,good_matches,img_good_match);
    imshow("all match points",img_match);
    imshow("good match points",img_good_match);
    waitKey(0);
    
    
    
    return 0;
}
