#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>
#include <cmath>
#include <iterator>
#include <opencv2/ml.hpp>
#include <experimental/filesystem> // C++-standard header file name
#include <filesystem> // Microsoft-specific implementation header file name
using namespace std::experimental::filesystem::v1;
using namespace cv::ml;
using namespace std;


// Sign must be at least 100x100
#define MINIMUM_SIGN_SIDE 100
#define MINIMUM_SIGN_AREA 10000
#define MINIMUM_SIGN_BOUNDARY_LENGTH 400
#define STANDARD_SIGN_WIDTH_AND_HEIGHT 200
// Best match must be 10% better than second best match
#define REQUIRED_RATIO_OF_BEST_TO_SECOND_BEST 1.1
// Located shape must overlap the ground truth by 80% to be considered a match
#define REQUIRED_OVERLAP 0.8

#define SVM_VECTOR_SIZE 9*6

class ObjectAndLocation
{ 
public:
	ObjectAndLocation(string object_name, Point top_left, Point top_right, Point bottom_right, Point bottom_left, Mat object_image);
	ObjectAndLocation(FileNode& node);
	void write(FileStorage& fs);
	void read(FileNode& node);
	Mat& getImage();
	string getName();
	void setName(string new_name);
	string getVerticesString();
	void DrawObject(Mat* display_image, Scalar& colour);
	double getMinimumSideLength();
	double getArea();
	void getVertice(int index, int& x, int& y);
	array<float, SVM_VECTOR_SIZE>& getSVM_vector();
	void setImage(Mat image);   // *** Student should add any initialisation (of their images or features; see private data below) they wish into this method.
	double compareObjects(ObjectAndLocation* otherObject);  // *** Student should write code to compare objects using chosen method.
	bool OverlapsWith(ObjectAndLocation* other_object);
private:
	string object_name;
	Mat image;
	vector<Point2i> vertices;
	array<float, SVM_VECTOR_SIZE> svm_vector;
	// *** Student can add whatever images or features they need to describe the object.
};

class AnnotatedImages;

class ImageWithObjects
{
	friend class AnnotatedImages;
public:
	ImageWithObjects(string passed_filename);
	ImageWithObjects(FileNode& node);
	virtual void LocateAndAddAllObjects(AnnotatedImages& training_images) = 0;
	ObjectAndLocation* addObject(string object_name, int top_left_column, int top_left_row, int top_right_column, int top_right_row,
		int bottom_right_column, int bottom_right_row, int bottom_left_column, int bottom_left_row, Mat& image);
	void write(FileStorage& fs);
	void read(FileNode& node);
	ObjectAndLocation* getObject(int index);
	void extractAndSetObjectImage(ObjectAndLocation *new_object);
	string ExtractObjectName(string filenamestr);
	void FindBestMatch(ObjectAndLocation* new_object, string& object_name, double& match_value);
protected:
	string filename;
	Mat image;
	vector<ObjectAndLocation> objects;
};

class ImageWithBlueSignObjects : public ImageWithObjects
{
public:
	ImageWithBlueSignObjects(string passed_filename);
	ImageWithBlueSignObjects(FileNode& node); 
	void LocateAndAddAllObjects(AnnotatedImages& training_images);  // *** Student needs to develop this routine and add in objects using the addObject method
};

class ConfusionMatrix;

class AnnotatedImages
{
public:
	AnnotatedImages(string directory_name);
	AnnotatedImages();
	void addAnnotatedImage(ImageWithObjects &annotated_image);
	void write(FileStorage& fs);
	void read(FileStorage& fs);
	void read(FileNode& node);
	void read(string filename);
	void LocateAndAddAllObjects(AnnotatedImages& training_images);
	void FindBestMatch(ObjectAndLocation* new_object);
	Mat getImageOfAllObjects(int break_after = 7);
	void CompareObjectsWithGroundTruth(AnnotatedImages& training_images, AnnotatedImages& ground_truth, ConfusionMatrix& results);
	ImageWithObjects* getAnnotatedImage(int index);
	ImageWithObjects* FindAnnotatedImage(string filename_to_find);
public:
	string name;
	vector<ImageWithObjects*> annotated_images;
};

class ConfusionMatrix
{
public:
	ConfusionMatrix(AnnotatedImages training_images);
	void AddMatch(string ground_truth, string recognised_as, bool duplicate = false);
	void AddFalseNegative(string ground_truth);
	void AddFalsePositive(string recognised_as);
	void Print();
private:
	void AddObjectClass(string object_class_name);
	int getObjectClassIndex(string object_class_name);
	vector<string> class_names;
	int confusion_size;
	int** confusion_matrix;
	int false_index;
	int tp, fp, fn;
};

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


ObjectAndLocation::ObjectAndLocation(string passed_object_name, Point top_left, Point top_right, Point bottom_right, Point bottom_left, Mat object_image)
{
	object_name = passed_object_name;
	vertices.push_back(top_left);
	vertices.push_back(top_right);
	vertices.push_back(bottom_right);
	vertices.push_back(bottom_left);
	setImage(object_image);
}
ObjectAndLocation::ObjectAndLocation(FileNode& node)
{
	read(node);
}
void ObjectAndLocation::write(FileStorage& fs)
{
	fs << "{" << "nameStr" << object_name;
	fs << "coordinates" << "[";
	for (int i = 0; i < vertices.size(); ++i)
	{
		fs << "[:" << vertices[i].x << vertices[i].y << "]";
	}
	fs << "]";
	fs << "}";
}
void ObjectAndLocation::read(FileNode& node)
{
	node["nameStr"] >> object_name;
	FileNode data = node["coordinates"];
	for (FileNodeIterator itData = data.begin(); itData != data.end(); ++itData)
	{
		// Read each point
		FileNode pt = *itData;

		Point2i point;
		FileNodeIterator itPt = pt.begin();
		point.x = *itPt; ++itPt;
		point.y = *itPt;
		vertices.push_back(point);
	}
}
Mat& ObjectAndLocation::getImage()
{
	return image;
}
array<float,SVM_VECTOR_SIZE>& ObjectAndLocation::getSVM_vector()
{
	return svm_vector;
}
string ObjectAndLocation::getName()
{
	return object_name;
}
void ObjectAndLocation::setName(string new_name)
{
	object_name.assign(new_name);
}
string ObjectAndLocation::getVerticesString()
{
	string result;
	for (int index = 0; (index < vertices.size()); index++)
		result.append("(" + to_string(vertices[index].x) + " " + to_string(vertices[index].y) + ") ");
	return result;
}
void ObjectAndLocation::DrawObject(Mat* display_image, Scalar& colour)
{
	writeText(*display_image, object_name, vertices[0].y - 8, vertices[0].x + 8, colour, 2.0, 4);
	polylines(*display_image, vertices, true, colour, 8);
}
double ObjectAndLocation::getMinimumSideLength()
{
	double min_distance = DistanceBetweenPoints(vertices[0], vertices[vertices.size() - 1]);
	for (int index = 0; (index < vertices.size() - 1); index++)
	{
		double distance = DistanceBetweenPoints(vertices[index], vertices[index + 1]);
		if (distance < min_distance)
			min_distance = distance;
	}
	return min_distance;
}
double ObjectAndLocation::getArea()
{
	return contourArea(vertices);
}
void ObjectAndLocation::getVertice(int index, int& x, int& y)
{
	if ((vertices.size() < index) || (index < 0))
		x = y = -1;
	else
	{
		x = vertices[index].x;
		y = vertices[index].y;
	}
}

ImageWithObjects::ImageWithObjects(string passed_filename)
{
	filename = strdup(passed_filename.c_str());
	cout << "Opening " << filename << endl;
	image = imread(filename, -1);
}
ImageWithObjects::ImageWithObjects(FileNode& node)
{
	read(node);
}
ObjectAndLocation* ImageWithObjects::addObject(string object_name, int top_left_column, int top_left_row, int top_right_column, int top_right_row,
	int bottom_right_column, int bottom_right_row, int bottom_left_column, int bottom_left_row, Mat& image)
{
	ObjectAndLocation new_object(object_name, Point(top_left_column, top_left_row), Point(top_right_column, top_right_row), Point(bottom_right_column, bottom_right_row), Point(bottom_left_column, bottom_left_row), image);
	objects.push_back(new_object);
	return &(objects[objects.size() - 1]);
}
void ImageWithObjects::write(FileStorage& fs)
{
	fs << "{" << "Filename" << filename << "Objects" << "[";
	for (int index = 0; index < objects.size(); index++)
		objects[index].write(fs);
	fs << "]" << "}";
}
void ImageWithObjects::extractAndSetObjectImage(ObjectAndLocation *new_object)
{
	Mat perspective_warped_image = Mat::zeros(STANDARD_SIGN_WIDTH_AND_HEIGHT, STANDARD_SIGN_WIDTH_AND_HEIGHT, image.type());
	Mat perspective_matrix(3, 3, CV_32FC1);
	int x[4], y[4];
	new_object->getVertice(0, x[0], y[0]);
	new_object->getVertice(1, x[1], y[1]);
	new_object->getVertice(2, x[2], y[2]);
	new_object->getVertice(3, x[3], y[3]);
	Point2f source_points[4] = { { ((float)x[0]), ((float)y[0]) },{ ((float)x[1]), ((float)y[1]) },{ ((float)x[2]), ((float)y[2]) },{ ((float)x[3]), ((float)y[3]) } };
	Point2f destination_points[4] = { { 0.0, 0.0 },{ STANDARD_SIGN_WIDTH_AND_HEIGHT - 1, 0.0 },{ STANDARD_SIGN_WIDTH_AND_HEIGHT - 1, STANDARD_SIGN_WIDTH_AND_HEIGHT - 1 },{ 0.0, STANDARD_SIGN_WIDTH_AND_HEIGHT - 1 } };
	perspective_matrix = getPerspectiveTransform(source_points, destination_points);
	warpPerspective(image, perspective_warped_image, perspective_matrix, perspective_warped_image.size());
	new_object->setImage(perspective_warped_image);
}
void ImageWithObjects::read(FileNode& node)
{
	filename = (string) node["Filename"];
	image = imread(filename, -1);
	FileNode images_node = node["Objects"];
	if (images_node.type() == FileNode::SEQ)
	{
		for (FileNodeIterator it = images_node.begin(); it != images_node.end(); ++it)
		{
			FileNode current_node = *it;
			ObjectAndLocation *new_object = new ObjectAndLocation(current_node);
			extractAndSetObjectImage(new_object);
			objects.push_back(*new_object);
		}
	}
}
ObjectAndLocation* ImageWithObjects::getObject(int index)
{
	if ((index < 0) || (index >= objects.size()))
		return NULL;
	else return &(objects[index]);
}
void ImageWithObjects::FindBestMatch(ObjectAndLocation* new_object, string& object_name, double& match_value)
{
	for (int index = 0; (index < objects.size()); index++)
	{
		double temp_match_score = objects[index].compareObjects(new_object);
		if ((temp_match_score > 0.0) && ((match_value < 0.0) || (temp_match_score < match_value)))
		{
			object_name = objects[index].getName();
			match_value = temp_match_score;
		}
	}
}

string ImageWithObjects::ExtractObjectName(string filenamestr)
{
	int last_slash = filenamestr.rfind("/");
	int start_of_object_name = (last_slash == std::string::npos) ? 0 : last_slash + 1;
	int extension = filenamestr.find(".", start_of_object_name);
	int end_of_filename = (extension == std::string::npos) ? filenamestr.length() - 1 : extension - 1;
	int end_of_object_name = filenamestr.find_last_not_of("1234567890", end_of_filename);
	end_of_object_name = (end_of_object_name == std::string::npos) ? end_of_filename : end_of_object_name;
	string object_name = filenamestr.substr(start_of_object_name, end_of_object_name - start_of_object_name + 1);
	return object_name;
}


ImageWithBlueSignObjects::ImageWithBlueSignObjects(string passed_filename) :
	ImageWithObjects(passed_filename)
{
}
ImageWithBlueSignObjects::ImageWithBlueSignObjects(FileNode& node) :
	ImageWithObjects(node)
{
}


AnnotatedImages::AnnotatedImages(string directory_name)
{
	name = directory_name;
	for (std::experimental::filesystem::directory_iterator next(std::experimental::filesystem::path(directory_name.c_str())), end; next != end; ++next)
	{
		read(next->path().generic_string());
	}
}
AnnotatedImages::AnnotatedImages()
{
	name = "";
}
void AnnotatedImages::addAnnotatedImage(ImageWithObjects &annotated_image)
{
	annotated_images.push_back(&annotated_image);
}

void AnnotatedImages::write(FileStorage& fs)
{
	fs << "AnnotatedImages";
	fs << "{";
	fs << "name" << name << "ImagesAndObjects" << "[";
	for (int index = 0; index < annotated_images.size(); index++)
		annotated_images[index]->write(fs);
	fs << "]" << "}";
}
void AnnotatedImages::read(FileStorage& fs)
{
	FileNode node = fs.getFirstTopLevelNode();
	read(node);
}
void AnnotatedImages::read(FileNode& node)
{
	name = (string)node["name"];
	FileNode images_node = node["ImagesAndObjects"];
	if (images_node.type() == FileNode::SEQ)
	{
		for (FileNodeIterator it = images_node.begin(); it != images_node.end(); ++it)
		{
			FileNode current_node = *it;
			ImageWithBlueSignObjects* new_image_with_objects = new ImageWithBlueSignObjects(current_node);
			annotated_images.push_back(new_image_with_objects);
		}
	}
}
void AnnotatedImages::read(string filename)
{
	ImageWithBlueSignObjects *new_image_with_objects = new ImageWithBlueSignObjects(filename);
	annotated_images.push_back(new_image_with_objects);
}
void AnnotatedImages::LocateAndAddAllObjects(AnnotatedImages& training_images)
{
	for (int index = 0; index < annotated_images.size(); index++)
	{
		annotated_images[index]->LocateAndAddAllObjects(training_images);
	}
}
void AnnotatedImages::FindBestMatch(ObjectAndLocation* new_object) //Mat& perspective_warped_image, string& object_name, double& match_value)
{
	double match_value = -1.0;
	string object_name = "Unknown";
	double temp_best_match = 1000000.0;
	string temp_best_name;
	double temp_second_best_match = 1000000.0;
	string temp_second_best_name;
	for (int index = 0; index < annotated_images.size(); index++)
	{
		annotated_images[index]->FindBestMatch(new_object, object_name, match_value);
		if (match_value < temp_best_match)
		{
			if (temp_best_name.compare(object_name) != 0)
			{
				temp_second_best_match = temp_best_match;
				temp_second_best_name = temp_best_name;
			}
			temp_best_match = match_value;
			temp_best_name = object_name;
		}
		else if ((match_value != temp_best_match) && (match_value < temp_second_best_match) && (temp_best_name.compare(object_name) != 0))
		{
			temp_second_best_match = match_value;
			temp_second_best_name = object_name;
		}
	}
	if (temp_second_best_match / temp_best_match < REQUIRED_RATIO_OF_BEST_TO_SECOND_BEST)
		new_object->setName("Unknown");
	else new_object->setName(temp_best_name);
}

Mat AnnotatedImages::getImageOfAllObjects(int break_after)
{
	Mat all_rows_so_far;
	Mat output;
	int count = 0;
	int object_index = 0;
	string blank("");
	for (int index = 0; (index < annotated_images.size()); index++)
	{
		ObjectAndLocation* current_object = NULL;
		int object_index = 0;
		while ((current_object = (annotated_images[index])->getObject(object_index)) != NULL)
		{
			if (count == 0)
			{
				output = JoinSingleImage(current_object->getImage(), current_object->getName());
			}
			else if (count % break_after == 0)
			{
				if (count == break_after)
					all_rows_so_far = output;
				else
				{
					Mat temp_rows = JoinImagesVertically(all_rows_so_far, blank, output, blank, 0);
					all_rows_so_far = temp_rows.clone();
				}
				output = JoinSingleImage(current_object->getImage(), current_object->getName());
			}
			else
			{
				Mat new_output = JoinImagesHorizontally(output, blank, current_object->getImage(), current_object->getName(), 0);
				output = new_output.clone();
			}
			count++;
			object_index++;
		}
	}
	if (count == 0)
	{
		Mat blank_output(1, 1, CV_8UC3, Scalar(0, 0, 0));
		return blank_output;
	}
	else if (count < break_after)
		return output;
	else {
		Mat temp_rows = JoinImagesVertically(all_rows_so_far, blank, output, blank, 0);
		all_rows_so_far = temp_rows.clone();
		return all_rows_so_far;
	}
}

ImageWithObjects* AnnotatedImages::getAnnotatedImage(int index)
{
	if ((index >= 0) && (index < annotated_images.size()))
		return annotated_images[index];
	else return NULL;
}

ImageWithObjects* AnnotatedImages::FindAnnotatedImage(string filename_to_find)
{
	for (int index = 0; (index < annotated_images.size()); index++)
	{
		if (filename_to_find.compare(annotated_images[index]->filename) == 0)
			return annotated_images[index];
	}
	return NULL;
}


bool PointInPolygon(Point2i point, vector<Point2i> vertices)
{
	int i, j, nvert = vertices.size();
	bool inside = false;

	for (i = 0, j = nvert - 1; i < nvert; j = i++)
	{
		if ((vertices[i].x == point.x) && (vertices[i].y == point.y))
			return true;
		if (((vertices[i].y >= point.y) != (vertices[j].y >= point.y)) &&
			(point.x <= (vertices[j].x - vertices[i].x) * (point.y - vertices[i].y) / (vertices[j].y - vertices[i].y) + vertices[i].x)
			)
			inside = !inside;
	}
	return inside;
}

bool ObjectAndLocation::OverlapsWith(ObjectAndLocation* other_object)
{
	double area = contourArea(vertices);
	double other_area = contourArea(other_object->vertices);
	double overlap_area = 0.0;
	int count_points_inside = 0;
	for (int index = 0; (index < vertices.size()); index++)
	{
		if (PointInPolygon(vertices[index], other_object->vertices))
			count_points_inside++;
	}
	int count_other_points_inside = 0;
	for (int index = 0; (index < other_object->vertices.size()); index++)
	{
		if (PointInPolygon(other_object->vertices[index], vertices))
			count_other_points_inside++;
	}
	if (count_points_inside == vertices.size())
		overlap_area = area;
	else if (count_other_points_inside == other_object->vertices.size())
		overlap_area = other_area;
	else if ((count_points_inside == 0) && (count_other_points_inside == 0))
		overlap_area = 0.0;
	else
	{   // There is a partial overlap of the polygons.
		// Find min & max x & y for the current object
		int min_x = vertices[0].x, min_y = vertices[0].y, max_x = vertices[0].x, max_y = vertices[0].y;
		for (int index = 0; (index < vertices.size()); index++)
		{
			if (min_x > vertices[index].x)
				min_x = vertices[index].x;
			else if (max_x < vertices[index].x)
				max_x = vertices[index].x;
			if (min_y > vertices[index].y)
				min_y = vertices[index].y;
			else if (max_y < vertices[index].y)
				max_y = vertices[index].y;
		}
		int min_x2 = other_object->vertices[0].x, min_y2 = other_object->vertices[0].y, max_x2 = other_object->vertices[0].x, max_y2 = other_object->vertices[0].y;
		for (int index = 0; (index < other_object->vertices.size()); index++)
		{
			if (min_x2 > other_object->vertices[index].x)
				min_x2 = other_object->vertices[index].x;
			else if (max_x2 < other_object->vertices[index].x)
				max_x2 = other_object->vertices[index].x;
			if (min_y2 > other_object->vertices[index].y)
				min_y2 = other_object->vertices[index].y;
			else if (max_y2 < other_object->vertices[index].y)
				max_y2 = other_object->vertices[index].y;
		}
		// We only need the maximum overlapping bounding boxes
		if (min_x < min_x2) min_x = min_x2;
		if (max_x > max_x2) max_x = max_x2;
		if (min_y < min_y2) min_y = min_y2;
		if (max_y > max_y2) max_y = max_y2;
		// For all points
		overlap_area = 0;
		Point2i current_point;
		// Try ever decreasing squares within the overlapping (image aligned) bounding boxes to find the overlapping area.
		bool all_points_inside = false;
		int distance_from_edge = 0;
		for (; ((distance_from_edge < (max_x - min_x + 1) / 2) && (distance_from_edge < (max_y - min_y + 1) / 2) && (!all_points_inside)); distance_from_edge++)
		{
			all_points_inside = true;
			for (current_point.x = min_x + distance_from_edge; (current_point.x <= (max_x - distance_from_edge)); current_point.x++)
				for (current_point.y = min_y + distance_from_edge; (current_point.y <= max_y - distance_from_edge); current_point.y += max_y - 2 * distance_from_edge - min_y)
				{
					if ((PointInPolygon(current_point, vertices)) && (PointInPolygon(current_point, other_object->vertices)))
						overlap_area++;
					else all_points_inside = false;
				}
			for (current_point.y = min_y + distance_from_edge + 1; (current_point.y <= (max_y - distance_from_edge - 1)); current_point.y++)
				for (current_point.x = min_x + distance_from_edge; (current_point.x <= max_x - distance_from_edge); current_point.x += max_x - 2 * distance_from_edge - min_x)
				{
					if ((PointInPolygon(current_point, vertices)) && (PointInPolygon(current_point, other_object->vertices)))
						overlap_area++;
					else all_points_inside = false;
				}
		}
		if (all_points_inside)
			overlap_area += (max_x - min_x + 1 - 2 * (distance_from_edge + 1)) * (max_y - min_y + 1 - 2 * (distance_from_edge + 1));
	}
	double percentage_overlap = (overlap_area*2.0) / (area + other_area);
	return (percentage_overlap >= REQUIRED_OVERLAP);
}



void AnnotatedImages::CompareObjectsWithGroundTruth(AnnotatedImages& training_images, AnnotatedImages& ground_truth, ConfusionMatrix& results)
{
	// For every annotated image in ground_truth, find the corresponding image in this
	for (int ground_truth_image_index = 0; ground_truth_image_index < ground_truth.annotated_images.size(); ground_truth_image_index++)
	{
		ImageWithObjects* current_annotated_ground_truth_image = ground_truth.annotated_images[ground_truth_image_index];
		ImageWithObjects* current_annotated_recognition_image = FindAnnotatedImage(current_annotated_ground_truth_image->filename);
		if (current_annotated_recognition_image != NULL)
		{
			ObjectAndLocation* current_ground_truth_object = NULL;
			int ground_truth_object_index = 0;
			Mat* display_image = NULL;
			if (!current_annotated_recognition_image->image.empty())
			{
				display_image = &(current_annotated_recognition_image->image);
			}
			// For each object in ground_truth.annotated_image
			while ((current_ground_truth_object = current_annotated_ground_truth_image->getObject(ground_truth_object_index)) != NULL)
			{
				if ((current_ground_truth_object->getMinimumSideLength() >= MINIMUM_SIGN_SIDE) &&
					(current_ground_truth_object->getArea() >= MINIMUM_SIGN_AREA))
				{
					// Determine the number of overlapping objects (correct & incorrect)
					vector<ObjectAndLocation*> overlapping_correct_objects;
					vector<ObjectAndLocation*> overlapping_incorrect_objects;
					ObjectAndLocation* current_recognised_object = NULL;
					int recognised_object_index = 0;
					// For each object in this.annotated_image
					while ((current_recognised_object = current_annotated_recognition_image->getObject(recognised_object_index)) != NULL)
					{
						if (current_recognised_object->getName().compare("Unknown") != 0)
							if (current_ground_truth_object->OverlapsWith(current_recognised_object))
							{
								if (current_ground_truth_object->getName().compare(current_recognised_object->getName()) == 0)
									overlapping_correct_objects.push_back(current_recognised_object);
								else overlapping_incorrect_objects.push_back(current_recognised_object);
							}
						recognised_object_index++;
					}
					if ((overlapping_correct_objects.size() == 0) && (overlapping_incorrect_objects.size() == 0))
					{
						if (display_image != NULL)
						{
							Scalar colour(0x00, 0x00, 0xFF);
							current_ground_truth_object->DrawObject(display_image, colour);
						}
						results.AddFalseNegative(current_ground_truth_object->getName());
						cout << current_annotated_ground_truth_image->filename << ", " << current_ground_truth_object->getName() << ", (False Negative) , " << current_ground_truth_object->getVerticesString() << endl;
					}
					else {
						for (int index = 0; (index < overlapping_correct_objects.size()); index++)
						{
							Scalar colour(0x00, 0xFF, 0x00);
							results.AddMatch(current_ground_truth_object->getName(), overlapping_correct_objects[index]->getName(), (index > 0));
							if (index > 0)
							{
								colour[2] = 0xFF;
								cout << current_annotated_ground_truth_image->filename << ", " << current_ground_truth_object->getName() << ", (Duplicate) , " << current_ground_truth_object->getVerticesString() << endl;
							}
							if (display_image != NULL)
								current_ground_truth_object->DrawObject(display_image, colour);
						}
						for (int index = 0; (index < overlapping_incorrect_objects.size()); index++)
						{
							if (display_image != NULL)
							{
								Scalar colour(0xFF, 0x00, 0xFF);
								overlapping_incorrect_objects[index]->DrawObject(display_image, colour);
							}
							results.AddMatch(current_ground_truth_object->getName(), overlapping_incorrect_objects[index]->getName(), (index > 0));
							cout << current_annotated_ground_truth_image->filename << ", " << current_ground_truth_object->getName() << ", (Mismatch), " << overlapping_incorrect_objects[index]->getName() << " , " << current_ground_truth_object->getVerticesString() << endl;;
						}
					}
				}
				else
					cout << current_annotated_ground_truth_image->filename << ", " << current_ground_truth_object->getName() << ", (DROPPED GT) , " << current_ground_truth_object->getVerticesString() << endl;

				ground_truth_object_index++;
			}
			//	For each object in this.annotated_image
			//				For each overlapping object in ground_truth.annotated_image
			//					Don't do anything (as already done above)
			//			If no overlapping objects.
			//				Update the confusion table (with a False Positive)
			ObjectAndLocation* current_recognised_object = NULL;
			int recognised_object_index = 0;
			// For each object in this.annotated_image
			while ((current_recognised_object = current_annotated_recognition_image->getObject(recognised_object_index)) != NULL)
			{
				if ((current_recognised_object->getMinimumSideLength() >= MINIMUM_SIGN_SIDE) &&
					(current_recognised_object->getArea() >= MINIMUM_SIGN_AREA))
				{
					// Determine the number of overlapping objects (correct & incorrect)
					vector<ObjectAndLocation*> overlapping_objects;
					ObjectAndLocation* current_ground_truth_object = NULL;
					int ground_truth_object_index = 0;
					// For each object in ground_truth.annotated_image
					while ((current_ground_truth_object = current_annotated_ground_truth_image->getObject(ground_truth_object_index)) != NULL)
					{
						if (current_ground_truth_object->OverlapsWith(current_recognised_object))
							overlapping_objects.push_back(current_ground_truth_object);
						ground_truth_object_index++;
					}
					if ((overlapping_objects.size() == 0) && (current_recognised_object->getName().compare("Unknown") != 0))
					{
						results.AddFalsePositive(current_recognised_object->getName());
						if (display_image != NULL)
						{
							Scalar colour(0x7F, 0x7F, 0xFF);
							current_recognised_object->DrawObject(display_image, colour);
						}
						cout << current_annotated_recognition_image->filename << ", " << current_recognised_object->getName() << ", (False Positive) , " << current_recognised_object->getVerticesString() << endl;
					}
				}
				else
					cout << current_annotated_recognition_image->filename << ", " << current_recognised_object->getName() << ", (DROPPED) , " << current_recognised_object->getVerticesString() << endl;
				recognised_object_index++;
			}
			if (display_image != NULL)
			{
				Mat smaller_image;
				resize(*display_image, smaller_image, Size(display_image->cols / 4, display_image->rows / 4));
				imshow(current_annotated_recognition_image->filename, smaller_image);
				char ch = cv::waitKey(1);
				//				delete display_image;
			}
		}
	}
}

// Determine object classes from the training_images (vector of strings)
// Create and zero a confusion matrix
ConfusionMatrix::ConfusionMatrix(AnnotatedImages training_images)
{
	// Extract object class names
	ImageWithObjects* current_annnotated_image = NULL;
	int image_index = 0;
	while ((current_annnotated_image = training_images.getAnnotatedImage(image_index)) != NULL)
	{
		ObjectAndLocation* current_object = NULL;
		int object_index = 0;
		while ((current_object = current_annnotated_image->getObject(object_index)) != NULL)
		{
			AddObjectClass(current_object->getName());
			object_index++;
		}
		image_index++;
	}
	// Create and initialise confusion matrix
	confusion_size = class_names.size() + 1;
	confusion_matrix = new int*[confusion_size];
	for (int index = 0; (index < confusion_size); index++)
	{
		confusion_matrix[index] = new int[confusion_size];
		for (int index2 = 0; (index2 < confusion_size); index2++)
			confusion_matrix[index][index2] = 0;
	}
	false_index = confusion_size - 1;
}
void ConfusionMatrix::AddObjectClass(string object_class_name)
{
	int index = getObjectClassIndex(object_class_name);
	if (index == -1)
		class_names.push_back(object_class_name);
	tp = fp = fn = 0;
}
int ConfusionMatrix::getObjectClassIndex(string object_class_name)
{
	int index = 0;
	for (; (index < class_names.size()) && (object_class_name.compare(class_names[index]) != 0); index++)
		;
	if (index < class_names.size())
		return index;
	else return -1;
}
void ConfusionMatrix::AddMatch(string ground_truth, string recognised_as, bool duplicate)
{
	if ((ground_truth.compare(recognised_as) == 0) && (duplicate))
		AddFalsePositive(recognised_as);
	else
	{
		confusion_matrix[getObjectClassIndex(ground_truth)][getObjectClassIndex(recognised_as)]++;
		if (ground_truth.compare(recognised_as) == 0)
			tp++;
		else {
			fp++;
			fn++;
		}
	}
}
void ConfusionMatrix::AddFalseNegative(string ground_truth)
{
	fn++;
	confusion_matrix[getObjectClassIndex(ground_truth)][false_index]++;
}
void ConfusionMatrix::AddFalsePositive(string recognised_as)
{
	fp++;
	confusion_matrix[false_index][getObjectClassIndex(recognised_as)]++;
}
void ConfusionMatrix::Print()
{
	cout << ",,,Recognised as:" << endl << ",,";
	for (int recognised_as_index = 0; recognised_as_index < confusion_size; recognised_as_index++)
		if (recognised_as_index < confusion_size - 1)
			cout << class_names[recognised_as_index] << ",";
		else cout << "False Negative,";
		cout << endl;
		for (int ground_truth_index = 0; (ground_truth_index <= class_names.size()); ground_truth_index++)
		{
			if (ground_truth_index < confusion_size - 1)
				cout << "Ground Truth," << class_names[ground_truth_index] << ",";
			else cout << "Ground Truth,False Positive,";
			for (int recognised_as_index = 0; recognised_as_index < confusion_size; recognised_as_index++)
				cout << confusion_matrix[ground_truth_index][recognised_as_index] << ",";
			cout << endl;
		}
		double precision = ((double)tp) / ((double)(tp + fp));
		double recall = ((double)tp) / ((double)(tp + fn));
		double f1 = 2.0*precision*recall / (precision + recall);
		cout << endl << "Precision = " << precision << endl << "Recall = " << recall << endl << "F1 = " << f1 << endl;
}

void MyApplication()
{
	AnnotatedImages trainingImages;
	FileStorage training_file("BlueSignsTraining.xml", FileStorage::READ);
	if (!training_file.isOpened())
	{
		cout << "Could not open the file: \"" << "BlueSignsTraining.xml" << "\"" << endl;
	}
	else
	{
		trainingImages.read(training_file);
	}
	training_file.release();
	Mat image_of_all_training_objects = trainingImages.getImageOfAllObjects();

//	imshow("All Training Objects", image_of_all_training_objects);
	imwrite("AllTrainingObjectImages.jpg", image_of_all_training_objects);

	// Train SVM Model
	int N = trainingImages.annotated_images.size(); // sample size
	vector<string> trainingLabels = {
		"Coffee",
		"Disabled",
		"Escalator",
		"Exit",
		"Gents",
		"Information",
		"Ladies",
		"Lift",
		"One",
		"Stairs",
		"TicketDesk",
		"Two"
	};

	array<int, 100> trainingset_labels = {};
	array<array<float, SVM_VECTOR_SIZE>, 100> trainingset_values = {};

	int count = 0;

	// for each annotated image
	for (int index = 0; index < N; index++)
	{
		ObjectAndLocation* current_object = NULL;
		int object_index = 0;
		while ((current_object = (trainingImages.annotated_images[index])->getObject(object_index)) != NULL)
		{
			vector<string>::iterator it = find(trainingLabels.begin(), trainingLabels.end(), current_object->getName());
			assert(it != trainingLabels.end());

			trainingset_labels.at(count) = distance(trainingLabels.begin(), it);
			cout << distance(trainingLabels.begin(), it) << " " << current_object->getName() << endl;

			trainingset_values.at(count) = current_object->getSVM_vector();

			object_index++;
			count++;
		}
	}

	Mat trainingDataMat(N, SVM_VECTOR_SIZE, CV_32F, trainingset_values.data());
	Mat labelsMat(N, 1, CV_32SC1, trainingset_labels.data());

	// Train the SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);		// n-class classification

	svm->setKernel(SVM::POLY);
	svm->setGamma(0.5);
	svm->setDegree(3);
	svm->setCoef0(6); 
	svm->setC(50);

	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);

	AnnotatedImages groundTruthImages;
	FileStorage ground_truth_file("BlueSignsGroundTruth.xml", FileStorage::READ);
	if (!ground_truth_file.isOpened())
	{
		cout << "Could not open the file: \"" << "BlueSignsGroundTruth.xml" << "\"" << endl;
	}
	else
	{
		groundTruthImages.read(ground_truth_file);
	}
	ground_truth_file.release();

	Mat image_of_all_ground_truth_objects = groundTruthImages.getImageOfAllObjects();
	imshow("All Ground Truth Objects", image_of_all_ground_truth_objects);
	imwrite("AllGroundTruthObjectImages.jpg", image_of_all_ground_truth_objects);

	AnnotatedImages unknownImages("Blue Signs/Testing");
	unknownImages.LocateAndAddAllObjects(trainingImages);

	// for each unknown image, find the corresponding label
	for (int index = 0; index < unknownImages.annotated_images.size(); index++)
	{
		ObjectAndLocation* current_object = NULL;
		int object_index = 0;
		while ((current_object = (unknownImages.annotated_images[index])->getObject(object_index)) != NULL)
		{
			array<float, SVM_VECTOR_SIZE> myobj = current_object->getSVM_vector();
			Mat SVM_vector(1, SVM_VECTOR_SIZE, CV_32F, myobj.data());

			float response = svm->predict(SVM_vector);

			string label = trainingLabels.at((int)response);
			current_object->setName(label);

			object_index++;
		}
	}

	FileStorage unknowns_file("BlueSignsTesting.xml", FileStorage::WRITE);
	if (!unknowns_file.isOpened())
	{
		cout << "Could not open the file: \"" << "BlueSignsTesting.xml" << "\"" << endl;
	}
	else
	{
		unknownImages.write(unknowns_file);
	}

	unknowns_file.release();

	Mat image_of_recognised_objects = unknownImages.getImageOfAllObjects();

	imshow("All Recognised Objects", image_of_recognised_objects);
	imwrite("AllRecognisedObjects.jpg", image_of_recognised_objects);

	ConfusionMatrix results(trainingImages);
	unknownImages.CompareObjectsWithGroundTruth(trainingImages, groundTruthImages, results);
	results.Print();
}


void ObjectAndLocation::setImage(Mat object_image)
{
	image = object_image.clone();
	if (image.size().width > 500 || image.size().height > 500) return;	// this is the original image 

	Mat my_sign = image.clone();
	resize(my_sign, my_sign, Size(450, 450));

	// crop image
	int offset_x = 35;
	int offset_y = 50;

	Rect crop(offset_x, offset_y, my_sign.size().width - (offset_x * 2), my_sign.size().height - (offset_y * 2));
	my_sign = my_sign(crop);

	Mat image_original = my_sign.clone();

	// to grayscale
	cvtColor(my_sign, my_sign, COLOR_BGR2GRAY);

	// Otsu thresholding
	threshold(my_sign, my_sign, 0, 255, THRESH_OTSU);

	// Closing
	morphologyEx(my_sign, my_sign, MORPH_CLOSE, Mat(2, 2, CV_8U, Scalar(1)));

	// Find contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	// Connected Component Analysis
	findContours(my_sign, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> approx(contours.size());
	vector<vector<Point>> keepContours;

	for (int i = 0; i < contours.size(); i++) {
		double peri = arcLength(contours[i], true);
		approxPolyDP(contours[i], approx[i], (double)(peri*0.006), true); // fit lines on contour

		if (approx[i].size() <= 4 || contourArea(approx[i]) < 1000) { // 4500
			// do nothing
		}
		else {
			keepContours.push_back(approx[i]);
		}
	}

	// sort by biggest area
	sort(keepContours.begin(), keepContours.end(), [](const vector<Point>& contour1, const vector<Point>& contour2)
	{
		return (contourArea(contour1) > contourArea(contour2));
	});

	vector<Point> hull(keepContours.size());

	vector<int> convexHull_IntIdx;
	vector<Vec4i> defects;

	svm_vector.fill(0);
	int svm_index = 0;

	// compute the following features: perimeter, area, bounding box, convex hull, ...
	for (int i = 0; i < 6; i++) {
		if (i < keepContours.size()) {
			Rect rect = boundingRect(keepContours[i]);

			convexHull(keepContours[i], hull, true);

			convexHull(keepContours[i], convexHull_IntIdx, false);
			convexityDefects(keepContours[i], convexHull_IntIdx, defects);
						
			// 9 features, SVM_VECTOR_SIZE
			float perimeter = arcLength(keepContours[i], true);
			float area = contourArea(keepContours[i]);
			float bounding_box = (float) rect.height *rect.width;
			float aspect_ratio = rect.height / (float)rect.width;
			float extent_ratio = (float)area / (rect.width*rect.height);
			float hull_ratio = (float) contourArea(hull) / area;
			float convex = isContourConvex(keepContours[i]);
			float convexitites_defect = defects.size();
			float circularity = (4 * PI*area) / (perimeter*perimeter);

			//svm_vector.insert(svm_vector.end(), { perimeter, area, aspect_ratio, extent_ratio, hull_ratio, convex });
			
			svm_vector[svm_index++] = perimeter;
			svm_vector[svm_index++] = area;
			svm_vector[svm_index++] = bounding_box;
			svm_vector[svm_index++] = aspect_ratio * 25;
			svm_vector[svm_index++] = extent_ratio * 50;
			svm_vector[svm_index++] = hull_ratio * 4500;
			svm_vector[svm_index++] = convex * 1000;
			svm_vector[svm_index++] = convexitites_defect * 100000;
			svm_vector[svm_index++] = circularity * 50;
		}
	}

	// *** Student should add any initialisation (of their images or features; see private data below) they wish into this method.
}


void ImageWithBlueSignObjects::LocateAndAddAllObjects(AnnotatedImages& training_images)
{
	string backprojection_sample("Backprojection/Sample.jpg");
	Mat blue_background_image = imread(backprojection_sample, -1);

	if(blue_background_image.empty() ) {
		cout << "Could not open " << backprojection_sample << endl;
		return;
	}

	// Resize the image
	Mat test_image;
	resize(image, test_image, cv::Size(image.cols * 0.3, image.rows * 0.3), 0, INTER_CUBIC );

	// Compute blue background histogramme for the back projection
	ColourHistogram histogram3D(blue_background_image, 5);
	histogram3D.NormaliseHistogram();

	// Back projection
	Mat back_projection_probabilities = histogram3D.BackProject(test_image);

	// Otsu thresholding
	threshold(back_projection_probabilities, back_projection_probabilities, 0, 255, THRESH_OTSU);

	// Erosion + Dilation
	morphologyEx(back_projection_probabilities, back_projection_probabilities, MORPH_OPEN, Mat(2, 2, CV_8U, Scalar(1)));

	//  Display the back projection image
	//imshow("Back Projection", back_projection_probabilities);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	// Connected Component Analysis
	findContours(back_projection_probabilities, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> approx(contours.size());;
	vector<vector<Point>> hull(contours.size());

	// Will be used to store our results
	vector<Rect> squares;
	vector<vector<Point>> sign_coordinates;

	// Parallelogram detection (sensitivity)
	int epsilon = 45;

	// Used for superposition
	Mat contoursImage = image.clone();

	// For each contour
	for (size_t i = 0; i < contours.size(); i++) {

		// reduce the number of vertices
		double peri = arcLength(contours[i], true);
		approxPolyDP(contours[i], approx[i], (double)(peri*0.1), true);

		// if the shape has 4 vertices, it is either a square or a rectangle
		if (approx[i].size() == 4) {

			// Let's fit the shape ...
			convexHull(approx[i], hull[i], true);
			vector<Point> hull_contours = hull[i];

			// ... we have 4 data points
			Point _1 = hull_contours[0];
			Point _2 = hull_contours[1];
			Point _3 = hull_contours[2];
			Point _4 = hull_contours[3];

			// Compute associated vectors 
			Point vector1(_2.x - _1.x, _2.y - _1.y);
			Point vector2(_3.x - _2.x, _3.y - _2.y);
			Point vector3(_4.x - _3.x, _4.y - _3.y);
			Point vector4(_1.x - _4.x, _1.y - _4.y);

			// We can now determine whether or not we have a parallelogram
			if (abs(vector1.x + vector3.x + vector3.y + vector1.y) + abs(vector2.x + vector4.x + vector4.y + vector2.y) <= epsilon) {

				// compute width & height
				double vector1_length = sqrt(pow(vector1.x, 2) + pow(vector1.y, 2));
				double vector2_length = sqrt(pow(vector2.x, 2) + pow(vector2.y, 2));

				// apply additional restrictions to the shape
				if (vector1_length > 40 && vector1_length <= 500 && vector2_length > 40 && vector2_length <= 500
					&& abs(vector1_length - vector2_length) <= 40) {

					// _1, _2, _3, _4 are not in the right order (random definition)
					// Let's determine the top_left, bottom_left, top_right, bottom_right points
					
					// For this we will find the two points on the extreme left (x_min1, x_min2) and on the extreme right (x_max1, x_max2)
					vector<Point> borders({ _1,_2,_3,_4 });

					Point x_min1, x_max1;
					int x_min = numeric_limits<int>::max();
					int x_max = numeric_limits<int>::min();

					for (int k = 0; k < borders.size(); k++) {
						if (borders[k].x < x_min) {
							x_min = borders[k].x;
							x_min1 = borders[k];
						}
						if (borders[k].x > x_max) {
							x_max = borders[k].x;
							x_max1 = borders[k];
						}
					}

					// remove x_min1 and x_max1 from vector<Point>, and compute x_min2 and x_max2
					borders.erase(find(borders.begin(), borders.end(), x_min1));
					borders.erase(find(borders.begin(), borders.end(), x_max1));

					Point x_min2, x_max2; 

					x_min = numeric_limits<int>::max();
					x_max = numeric_limits<int>::min();

					for (int k = 0; k < borders.size(); k++) {
						if (borders[k].x < x_min) {
							x_min = borders[k].x;
							x_min2 = borders[k];
						}
						if (borders[k].x > x_max) {
							x_max = borders[k].x;
							x_max2 = borders[k];
						}
					}

					// Readjust the points accordingly
					Point top_left = (x_min1.y < x_min2.y) ? x_min1 : x_min2;
					Point bottom_left = (x_min1.y > x_min2.y) ? x_min1 : x_min2;
					Point top_right = (x_max1.y < x_max2.y) ? x_max1 : x_max2;
					Point bottom_right = (x_max1.y > x_max2.y) ? x_max1 : x_max2;

					Rect detected_sign(top_left, bottom_right);
					squares.push_back(detected_sign);

					sign_coordinates.push_back(vector<Point>({ top_left,top_right,bottom_right, bottom_left }));
				}
			}

			/* **************************
				 SMALLER SHAPE DETECTION
			**************************************/

			// Compute the bounding box of the contour and use the bounding box to compute the aspect ratio
			Rect box = boundingRect(approx[i]);
			double ar = box.width / float(box.height);

			// check if it fills the bounding box
			double countour_area = contourArea(contours[i]);
			double box_area = box.width*box.height;

			double box_match = countour_area / box_area; // in [0,1]

			// a square will have an aspect ratio that is approximately
			// equal to one, otherwise, the shape is a rectangle
			if (box.width >= 10 && box.width <= 500 && box_match > 0.9)
			{
				//	Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
				//	rectangle(contoursImage, box.tl(), box.br(), color, 2, 8, 0);

				sign_coordinates.push_back(vector<Point>({ Point(box.x, box.y), Point(box.x + box.width, box.y), 
					Point(box.x + box.width, box.y + box.height), Point(box.x, box.y + box.height) }));

				squares.push_back(box);
			}
		}
	}

	// Merge overlapping rectangle
	for (int i = 0; i < squares.size(); i++) {
		Point centroid1 = Point(squares[i].x + squares[i].width / 2.0, squares[i].y + squares[i].height / 2.0);

		for (int j = 0; j < squares.size(); j++) {

			// Merge children into their parent
			Point centroid2 = Point(squares[j].x + squares[j].width / 2.0, squares[j].y + squares[j].height / 2.0);

			if (centroid2.x > squares[i].x && centroid2.x < squares[i].x + squares[i].width
				&& centroid2.y > squares[i].y && centroid2.y < squares[i].y + squares[i].height)
			{
				// rect[j] is inside i, remove rect[j]
				squares.erase(squares.begin() + j);
				sign_coordinates.erase(sign_coordinates.begin() + j);
				continue;
			}

			if (centroid1.x > squares[j].x && centroid1.x < squares[j].x + squares[j].width
				&& centroid1.y > squares[i].y && centroid1.y < squares[j].y + squares[j].height)
			{
				// rect[i] is inside j, remove rect[i]
				squares.erase(squares.begin() + i);
				sign_coordinates.erase(sign_coordinates.begin() + i);
				break;
			}

			Rect intersect = squares[i] & squares[j];
			if (intersect.area() > 0) // The two rectangle should be merged
			{
				if (intersect.area() == squares[j].area())
				{
					// rect[i] is inside j, remove rect[i]
					squares.erase(squares.begin() + i);
					sign_coordinates.erase(sign_coordinates.begin() + i);
					break;

				}
				else if (intersect.area() == squares[i].area())
				{
					// rect[j] is inside i, remove rect[j]
					squares.erase(squares.begin() + j);
					sign_coordinates.erase(sign_coordinates.begin() + j);
					continue;
				}
			}

		}
	}

	int count = 0;
	for (int i = 0; i < squares.size(); i++) {
		double ar = squares[i].width / float(squares[i].height);

		if (ar >= 0.45 && ar <= 1.65 && squares[i].width > 20) {

			Scalar color(0xFF, 0x00, 0xFF);
			Mat sign_hls, hsl_binary;

			// Check if it is actually a sign (using some heuristics)

			// Use perspective transformation
			Mat perspective_warped_image = Mat::zeros(STANDARD_SIGN_WIDTH_AND_HEIGHT, STANDARD_SIGN_WIDTH_AND_HEIGHT, image.type());
			Mat perspective_matrix(3, 3, CV_32FC1);
			int x[4], y[4];
			Point2f source_points[4] = { Point2f((float)(sign_coordinates[i][0].x / 0.3), (float)(sign_coordinates[i][0].y / 0.3)),
				Point2f((float)(sign_coordinates[i][1].x / 0.3), (float)(sign_coordinates[i][1].y / 0.3)),
				Point2f((float)(sign_coordinates[i][2].x / 0.3), (float)(sign_coordinates[i][2].y / 0.3)),
				Point2f((float)(sign_coordinates[i][3].x / 0.3), (float)(sign_coordinates[i][3].y / 0.3))
			};
			Point2f destination_points[4] = { { 0.0, 0.0 },{ STANDARD_SIGN_WIDTH_AND_HEIGHT - 1, 0.0 },{ STANDARD_SIGN_WIDTH_AND_HEIGHT - 1, STANDARD_SIGN_WIDTH_AND_HEIGHT - 1 },{ 0.0, STANDARD_SIGN_WIDTH_AND_HEIGHT - 1 } };
			perspective_matrix = getPerspectiveTransform(source_points, destination_points);
			warpPerspective(image, perspective_warped_image, perspective_matrix, perspective_warped_image.size());

			// convert to HSL
			cvtColor(perspective_warped_image, sign_hls, COLOR_RGB2HLS);

			// retrieve the luminance channel ...
			vector<Mat> hslChannels;
			split(sign_hls, hslChannels);

			// and use a threshold to get the brightest pixels
			threshold(hslChannels[2], hsl_binary, 70, 255, THRESH_BINARY);
			Scalar ratio = sum(hsl_binary) / (hsl_binary.size().width * hsl_binary.size().height);

			// add sign
			if (ratio[0] > 100) {
				ObjectAndLocation * add_sign = addObject(string("Unknown"), sign_coordinates[i][0].x / 0.3, sign_coordinates[i][0].y / 0.3, sign_coordinates[i][1].x / 0.3,
					sign_coordinates[i][1].y / 0.3, sign_coordinates[i][2].x / 0.3, sign_coordinates[i][2].y / 0.3, sign_coordinates[i][3].x / 0.3, sign_coordinates[i][3].y / 0.3, image);

				extractAndSetObjectImage(add_sign);
				count++;
			}
		}
	}

	cout << "Added " << count << " squares to " << filename << endl;

	// *** Student needs to develop this routine and add in objects using the addObject method
}


#define BAD_MATCHING_VALUE 1000000000.0;
double ObjectAndLocation::compareObjects(ObjectAndLocation* otherObject)
{
	// *** Student should write code to compare objects using chosen method.
	// Please bear in mind that ImageWithObjects::FindBestMatch assumes that the lower the value the better.  Feel free to change this.
	return BAD_MATCHING_VALUE;
}
