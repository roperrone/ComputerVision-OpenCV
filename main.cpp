/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <list>
#include <opencv2/ml.hpp>

#define REGENTHOUSE_IMAGE_INDEX 0
#define CAMPANILE1_IMAGE_INDEX 1
#define CAMPANILE2_IMAGE_INDEX 2
#define CREST_IMAGE_INDEX 3
#define OLDLIBRARY_IMAGE_INDEX 4
#define WINDOW1_IMAGE_INDEX 5
#define WINDOW2_IMAGE_INDEX 6
#define WINDOW1_LOCATIONS_IMAGE_INDEX 7
#define WINDOW2_LOCATIONS_IMAGE_INDEX 8
#define BIKES_IMAGE_INDEX 9
#define PEOPLE2_IMAGE_INDEX 10
#define ASTRONAUT_IMAGE_INDEX 11
#define PEOPLE1_IMAGE_INDEX 12
#define PEOPLE1_SKIN_MASK_IMAGE_INDEX 13
#define SKIN_IMAGE_INDEX 14
#define CHURCH_IMAGE_INDEX 15
#define FRUIT_IMAGE_INDEX 16
#define COATS_IMAGE_INDEX 17
#define STATIONARY_IMAGE_INDEX 18
#define PETS124_IMAGE_INDEX 19
#define PETS129_IMAGE_INDEX 20
#define PCB_IMAGE_INDEX 21
#define LICENSE_PLATE_IMAGE_INDEX 22
#define BICYCLE_BACKGROUND_IMAGE_INDEX 23
#define BICYCLE_MODEL_IMAGE_INDEX 24
#define NUMBERS_IMAGE_INDEX 25
#define GOOD_ORINGS_IMAGE_INDEX 26
#define BAD_ORINGS_IMAGE_INDEX 27
#define UNKNOWN_ORINGS_IMAGE_INDEX 28

#define SURVEILLANCE_VIDEO_INDEX 0
#define BICYCLES_VIDEO_INDEX 1
#define ABANDONMENT_VIDEO_INDEX 2

#define HAAR_FACE_CASCADE_INDEX 0

class Histogram
{
protected:
	Mat mImage;
	int mNumberChannels;
	int* mChannelNumbers;
	int* mNumberBins;
	float mChannelRange[2];
public:
	Histogram(Mat image, int number_of_bins)
	{
		mImage = image;
		mNumberChannels = mImage.channels();
		mChannelNumbers = new int[mNumberChannels];
		mNumberBins = new int[mNumberChannels];
		mChannelRange[0] = 0.0;
		mChannelRange[1] = 255.0;
		for (int count = 0; count < mNumberChannels; count++)
		{
			mChannelNumbers[count] = count;
			mNumberBins[count] = number_of_bins;
		}
		//ComputeHistogram();
	}
	virtual void ComputeHistogram() = 0;
	virtual void NormaliseHistogram() = 0;
	static void Draw1DHistogram(MatND histograms[], int number_of_histograms, Mat& display_image)
	{
		int number_of_bins = histograms[0].size[0];
		double max_value = 0, min_value = 0;
		double channel_max_value = 0, channel_min_value = 0;
		for (int channel = 0; (channel < number_of_histograms); channel++)
		{
			minMaxLoc(histograms[channel], &channel_min_value, &channel_max_value, 0, 0);
			max_value = ((max_value > channel_max_value) && (channel > 0)) ? max_value : channel_max_value;
			min_value = ((min_value < channel_min_value) && (channel > 0)) ? min_value : channel_min_value;
		}
		float scaling_factor = ((float)256.0) / ((float)number_of_bins);

		Mat histogram_image((int)(((float)number_of_bins)*scaling_factor) + 1, (int)(((float)number_of_bins)*scaling_factor) + 1, CV_8UC3, Scalar(255, 255, 255));
		display_image = histogram_image;
		line(histogram_image, Point(0, 0), Point(0, histogram_image.rows - 1), Scalar(0, 0, 0));
		line(histogram_image, Point(histogram_image.cols - 1, histogram_image.rows - 1), Point(0, histogram_image.rows - 1), Scalar(0, 0, 0));
		int highest_point = static_cast<int>(0.9*((float)number_of_bins)*scaling_factor);
		for (int channel = 0; (channel < number_of_histograms); channel++)
		{
			int last_height;
			for (int h = 0; h < number_of_bins; h++)
			{
				float value = histograms[channel].at<float>(h);
				int height = static_cast<int>(value*highest_point / max_value);
				int where = (int)(((float)h)*scaling_factor);
				if (h > 0)
					line(histogram_image, Point((int)(((float)(h - 1))*scaling_factor) + 1, (int)(((float)number_of_bins)*scaling_factor) - last_height),
						Point((int)(((float)h)*scaling_factor) + 1, (int)(((float)number_of_bins)*scaling_factor) - height),
						Scalar(channel == 0 ? 255 : 0, channel == 1 ? 255 : 0, channel == 2 ? 255 : 0));
				last_height = height;
			}
		}
	}
};

class ColourHistogram : public Histogram
{
private:
	MatND mHistogram;
public:
	ColourHistogram(Mat all_images[], int number_of_images, int number_of_bins) :
		Histogram(all_images[0], number_of_bins)
	{
		const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
		for (int index = 0; index < number_of_images; index++)
			calcHist(&mImage, 1, mChannelNumbers, Mat(), mHistogram, mNumberChannels, mNumberBins, channel_ranges, true, true);
	}
	ColourHistogram(Mat image, int number_of_bins) :
		Histogram(image, number_of_bins)
	{
		ComputeHistogram();
	}
	void ComputeHistogram()
	{
		const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
		calcHist(&mImage, 1, mChannelNumbers, Mat(), mHistogram, mNumberChannels, mNumberBins, channel_ranges);
	}
	void NormaliseHistogram()
	{
		normalize(mHistogram, mHistogram, 1.0);
	}
	Mat BackProject(Mat& image)
	{
		Mat& result = image.clone();
		const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
		calcBackProject(&image, 1, mChannelNumbers, mHistogram, result, channel_ranges, 255.0);
		return result;
	}
	MatND getHistogram()
	{
		return mHistogram;
	}
};


int main(int argc, const char** argv)
{
	MyApplication()	
		
} 

