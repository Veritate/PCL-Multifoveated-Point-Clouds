#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>

#include "foveatedCloud.h"
#include <pcl/common/centroid.h>
#include <pcl/common/time.h>
//#include <omp.h>

//typedef pcl::PointXYZRGBA PointType;
typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

std::string model_filename_;
std::string scene_filename_;

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool show_background_white_ (false);
bool use_viewer_ (false); // Set 'true' to execute this code and visualize the result in your local machine.
bool use_fovea (false);
int num_fovea (1);
float model_ss_ (0.01f);
float scene_ss_ (0.03f);
float rf_rad_ (0.015f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);
float cg_thresh_ (5.0f);
float foveaDistance (0.5f);
float foveaFactor (0);


FoveatedCloud<PointType> *foveatedScene;
FoveatedCloud<PointType> *foveatedKeypoints;
pcl::visualization::PCLVisualizer *viewer;

void showHelp (char *filename) {
	std::cout << std::endl;
	std::cout << "***************************************************************************" << std::endl;
	std::cout << "*                                                                         *" << std::endl;
	std::cout << "*             Correspondence Grouping Tutorial - Usage Guide              *" << std::endl;
	std::cout << "*                                                                         *" << std::endl;
	std::cout << "***************************************************************************" << std::endl << std::endl;
	std::cout << "Usage: " << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl << std::endl;
	std::cout << "Options:" << std::endl;
	std::cout << "     -h:                     Show this help." << std::endl;
	std::cout << "     -k:                     Show used keypoints." << std::endl;
	std::cout << "     -c:                     Show used correspondences." << std::endl;
	std::cout << "     -b:                     Set background white." << std::endl;
	std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
	std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
	std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
	std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
	std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
	std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
	std::cout << "     --m val:                Number of levels - 1 (default 1)" << std::endl << std::endl;
	std::cout << "     --min_res val:           Size of the lowest leaf size (default 0.01)" << std::endl << std::endl;
	std::cout << "     --max_res val:           Size of the greatest leaf size (default 0.001)" << std::endl << std::endl;
	std::cout << "     --foveafactor val:      Foveated growth factor (default 0)" << std::endl << std::endl;
	std::cout << "     --use_viewer (0|1):     If the visualizer is opened" << std::endl << std::endl;
	std::cout << "     --use_fovea (0|1):      If foveation is applied or not" << std::endl << std::endl;
	std::cout << "     --num_fovea (n>0):      Set the number of fovea" << std::endl << std::endl;
	std::cout << "     --r_m val:           Min foveated keypoints uniform sampling radius (default 0.01)" << std::endl << std::endl;
	std::cout << "     --r_0 val:           Max foveated keypoints uniform sampling radius (default 0.1)" << std::endl << std::endl;
}

void parseCommandLine (int argc, char *argv[]) {
	//Show help
	if (pcl::console::find_switch (argc, argv, "-h")) {
		showHelp (argv[0]);
		exit (0);
	}

	//Model & scene filenames
	std::vector<int> filenames;
	filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
	if (filenames.size () != 2) {
		std::cout << "Filenames missing.\n";
		showHelp (argv[0]);
		exit (-1);
	}

	model_filename_ = argv[filenames[0]];
	scene_filename_ = argv[filenames[1]];

	//General parameters
	pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
	pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss_);
	pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
	pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
	pcl::console::parse_argument (argc, argv, "--cg_size", cg_size_);
	pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
	pcl::console::parse_argument (argc, argv, "--use_viewer", use_viewer_);
	pcl::console::parse_argument (argc, argv, "--use_fovea", use_fovea);	

	//Visualization behavior
	if(use_viewer_) {
		if (pcl::console::find_switch (argc, argv, "-k"))
			show_keypoints_ = true;
		if (pcl::console::find_switch (argc, argv, "-c"))
			show_correspondences_ = true;
		if (pcl::console::find_switch (argc, argv, "-b"))
			show_background_white_ = true;
	}

	
	// Parameters of fovea approach only
	if(use_fovea)
	{
	       
	        int m;
		pcl::console::parse_argument (argc, argv, "--num_fovea", num_fovea);
		num_fovea = num_fovea > 0? num_fovea: 1;
	        foveatedScene = new FoveatedCloud<PointType>(num_fovea);
	        foveatedKeypoints = new FoveatedCloud<PointType>(num_fovea);
	        foveatedScene->fovea_factor = foveaFactor;
                foveatedKeypoints->fovea_factor = foveaFactor;
		
		pcl::console::parse_argument (argc, argv, "--foveafactor", foveaFactor);
	        pcl::console::parse_argument (argc, argv, "--m", m);
		foveatedScene->setM(m);
		foveatedKeypoints->setM(m);
	        pcl::console::parse_argument (argc, argv, "--min_res", foveatedScene->min_res);
	        pcl::console::parse_argument (argc, argv, "--max_res", foveatedScene->max_res);		
		
 	        std::vector<float> floatVec;

		int fovea_coord_total = 3*num_fovea;

		if(pcl::console::parse_x_arguments(argc, argv, "--offset", floatVec) > -1) {		    
		  if(floatVec.size() == fovea_coord_total) {
		    for(int i=0; i < num_fovea; i++)
		      foveatedScene->setStartCoord(floatVec[3*i], floatVec[3*i + 1], floatVec[3*i + 2], i);
		  } else {
		    std::cout << "Offset must be provided using --offset x,y,z.\n";
		  }
		} else {
		  std::cout << "An offset must be provided.\n";
		  showHelp (argv[0]);
		  exit (-1);
		}
		if(pcl::console::parse_x_arguments(argc, argv, "--u", floatVec) > -1) {
		  if(floatVec.size() == fovea_coord_total) {
		    for(int i=0; i < num_fovea; i++)
		      foveatedScene->setModelSize(floatVec[3*i], floatVec[3*i + 1], floatVec[3*i + 2], i);
		  } else {
		    std::cout << "Foveated model limits must be provided using --u x,y,z.\n";
		  }
		} else {
		  std::cout << "A foveated model limits must be provided.\n";
		  showHelp (argv[0]);
		  exit (-1);
		}
		if(pcl::console::parse_x_arguments(argc, argv, "--w", floatVec) > -1) {
		  if(floatVec.size() == fovea_coord_total) {
		    for(int i=0; i < num_fovea; i++)
		      foveatedScene->setFoveatedLevelSize(floatVec[3*i], floatVec[3*i + 1], floatVec[3*i + 2], i);
		  } else {
		    std::cout << "Foveated level limits must be provided using --w x,y,z.\n";
		  }
		} else {
		  std::cout << "A foveated level limits must be provided.\n";
		  showHelp (argv[0]);
		  exit (-1);
		}
		if(pcl::console::parse_x_arguments(argc, argv, "--f", floatVec) > -1) {
		  if(floatVec.size() == fovea_coord_total) {
		    for(int i=0; i < num_fovea; i++)
		      foveatedScene->setFovea(floatVec[3*i], floatVec[3*i + 1], floatVec[3*i + 2], i);
		  } else {
		    std::cout << "Fovea must be provided using --f x,y,z.\n";
		  }
		} else {
		  std::cout << "Warning: fovea not provided\n";
		  foveatedScene->setFovea(0, 0, 0);
		}
		foveatedScene->copyParameters(*foveatedKeypoints);
		foveatedKeypoints->initBoundariesFovea();
		pcl::console::parse_argument (argc, argv, "--r_m", foveatedKeypoints->min_res);
		pcl::console::parse_argument (argc, argv, "--r_0", foveatedKeypoints->max_res);	
	}
}


int main (int argc, char *argv[]) 
{
	// Read the parameters passed by OPTIONS variable
	parseCommandLine (argc, argv);

	// Clocks used to measure module and total computation times
	pcl::StopWatch clock;
	pcl::StopWatch clockTotal;
	
	// Point clouds declaration	
	pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
	pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
	pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
	pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());
	// The Final point cloud - concatenate the scene model and the computed scene_keypoints
	pcl::PointCloud<PointType>::Ptr final (new pcl::PointCloud<PointType> ());

	//Alignment axis
	pcl::PointCloud<PointType>::Ptr  ground (new pcl::PointCloud<PointType> ()); 
	pcl::PointCloud<PointType>::Ptr  centeredcloud (new pcl::PointCloud<PointType> ()); 
	pcl::PointCloud<PointType>::Ptr  groundTransformed (new pcl::PointCloud<PointType> ());	
	//
	
	//  Load clouds
	//
	std::cout << std::endl;
	std::cout << "-- Module Computation Time: " << std::endl;
	clock.reset();
	if (pcl::io::loadPCDFile (model_filename_, *model) < 0) {
		std::cout << "Error loading model cloud." << std::endl;
		showHelp (argv[0]);
		return (-1);
	}
	std::cout << std::endl;
	std::cout << "\tLoading model: " << clock.getTimeSeconds() << "s" << std::endl;

	clock.reset();
	if (pcl::io::loadPCDFile (scene_filename_, *scene) < 0) {
		std::cout << "Error loading scene cloud." << std::endl;
		showHelp (argv[0]);
		return (-1);
	}
	std::cout << "\tLoading scene: " << clock.getTimeSeconds() <<  "s" <<  std::endl;

	
	//
	//  Compute Model Normals 
	//
	clock.reset();
	pcl::NormalEstimationOMP<PointType, NormalType> norm_est;

	norm_est.setKSearch (10);
	norm_est.setInputCloud (model);
	norm_est.compute (*model_normals);

	std::cout << "\tModel Normals: " << clock.getTimeSeconds() <<  "s" <<  std::endl;


	//
	//  Downsample Model Cloud to Extract keypoints
	//
	pcl::PointCloud<int> sampled_indices;

	clock.reset();
	pcl::UniformSampling<PointType> uniform_sampling;
	uniform_sampling.setInputCloud (model);
	uniform_sampling.setRadiusSearch (model_ss_);	
	//uniform_sampling.compute (sampled_indices);	
	pcl::copyPointCloud (*model, sampled_indices.points, *model_keypoints);

	printf("\tModel Keypoints: %.03fs\n", clock.getTimeSeconds());
	uniform_sampling.filter (*model_keypoints); //Teste
	//
	//  Compute Descriptor for Model Keypoints
	//
	clock.reset();
	pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
	descr_est.setRadiusSearch (descr_rad_);
	descr_est.setInputCloud (model_keypoints);
	descr_est.setInputNormals (model_normals);
	descr_est.setSearchSurface (model);
	descr_est.compute (*model_descriptors);
	printf("\tModel Keypoints Descriptors: %.03fs\n", clock.getTimeSeconds());
	
	//
	// Initialize clock total ============================================================================
	//
	clockTotal.reset();

	// 
	// Apply Multiresolution to Scene Cloud
	// Insert foveated levels
	// 
	if(use_fovea) {
	        clock.reset();
		foveatedScene->clear();
		foveatedScene->genFoveasBoundaries();///		
		for(int j = 0; j < num_fovea; j++)
		  for(int i = 0; i <= foveatedScene->getM(); i++)	    
		    foveatedScene->addCloudLevel3(scene, i, j);
		printf("Scene size -> %lu\n", scene->size());
		printf("Foveated scene point size-> %lu\n", foveatedScene->points->size());
		scene->clear();
		*scene = *foveatedScene->points;
		foveatedKeypoints->method = FOVEATED_UNIFORMSAMPLING;
		foveatedKeypoints->copyFoveasBoundaries(foveatedScene->getDeltax(), foveatedScene->getDeltay(), foveatedScene->getDeltaz(), foveatedScene->getSkx(), foveatedScene->getSky(), foveatedScene->getSkz());
		for(int j = 0; j < num_fovea; j++)
		  for(int i = 0; i <= foveatedScene->getM(); i++)
		    foveatedKeypoints->addCloudLevel3(scene, i, j);
		printf("\tMultiresolution (Foveation): %.03fs\n", clock.getTimeSeconds());
	}

	//
	//  Compute Scene Normals 
	//
	clock.reset();
	norm_est.setKSearch (10);
	norm_est.setInputCloud (scene);
	norm_est.compute (*scene_normals);
	printf("\tScene Normals : %.03fs\n", clock.getTimeSeconds());

	//
	//  Downsample Scene Cloud to Extract keypoints
	//
	clock.reset();
	scene_keypoints->clear();
	if(use_fovea) {
		for(int i = 0; i < foveatedKeypoints->points->size(); i++) {
			scene_keypoints->push_back(foveatedKeypoints->points->at(i));
		}
	} else {
		uniform_sampling.setInputCloud (scene);
		uniform_sampling.setRadiusSearch (scene_ss_);
		//uniform_sampling.compute (sampled_indices);
		pcl::copyPointCloud (*scene, sampled_indices.points, *scene_keypoints);
	        uniform_sampling.filter (*scene_keypoints);
	}
	printf("\tScene Keypoints: %.03fs\n", clock.getTimeSeconds());

	//
	//  Compute Descriptor for Scene Keypoints
	//
	clock.reset();
	pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_estScene;
	descr_estScene.setRadiusSearch (descr_rad_);
	descr_estScene.setInputCloud (scene_keypoints);
	descr_estScene.setInputNormals (scene_normals);
	descr_estScene.setSearchSurface (scene);
	descr_estScene.compute (*scene_descriptors);
	printf("\tScene Keypoints Descriptors: %.03fs\n", clock.getTimeSeconds());

	//
	//  Find Model-Scene Correspondences with KdTree
	//
	clock.reset();
	pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

	pcl::KdTreeFLANN<DescriptorType> match_search;
	match_search.setInputCloud (model_descriptors);

	//
	//  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
	//
	for (size_t i = 0; i < scene_descriptors->size (); ++i)
	{
		std::vector<int> neigh_indices (1);
		std::vector<float> neigh_sqr_dists (1);
		if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
		{
			continue;
		}
		int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
		if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
		{
			pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
			model_scene_corrs->push_back (corr);
		}
	}
	printf("\tCorrespondences: %.03fs\n", clock.getTimeSeconds());

	clock.reset();
	//
	//  Actual Clustering
	//
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
	std::vector<pcl::Correspondences> clustered_corrs;

	//
	//  Using Hough3D
	//  Compute (Keypoints) Reference Frames
	//
	pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
	pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

	pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
	rf_est.setFindHoles (true);
	rf_est.setRadiusSearch (rf_rad_);

	rf_est.setInputCloud (model_keypoints);
	rf_est.setInputNormals (model_normals);
	rf_est.setSearchSurface (model);
	rf_est.compute (*model_rf);

	rf_est.setInputCloud (scene_keypoints);
	rf_est.setInputNormals (scene_normals);
	rf_est.setSearchSurface (scene);
	rf_est.compute (*scene_rf);

	//  Clustering
	pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
	clusterer.setHoughBinSize (cg_size_);
	clusterer.setHoughThreshold (cg_thresh_);
	clusterer.setUseInterpolation (true);
	clusterer.setUseDistanceWeight (false);

	clusterer.setInputCloud (model_keypoints);
	clusterer.setInputRf (model_rf);
	clusterer.setSceneCloud (scene_keypoints);
	clusterer.setSceneRf (scene_rf);
	clusterer.setModelSceneCorrespondences (model_scene_corrs);

	 //clusterer.cluster (clustered_corrs);
	clusterer.recognize (rototranslations, clustered_corrs);

	printf("\tClustering: %.03fs\n", clock.getTimeSeconds());

	std::cout << std::endl;
  	std::cout << " ------------------------------------------" << std::endl;
	std::cout << " \tTotal computation time: " << clockTotal.getTimeSeconds() << "s" << std::endl;
	std::cout << " ------------------------------------------" << std::endl << std::endl;

	
	//
	//  Output results
	//
	std::cout << std::endl;
	std::cout << "-- Processing informations" << std::endl;
	std::cout << "\tModel total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
	std::cout << "\tScene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;
	std::cout << "\tCorrespondences found: " << model_scene_corrs->size () << std::endl;
	std::cout << "\tModel instances found: " << rototranslations.size () << std::endl;
	std::cout << std::endl;
	
	// Append the scene cloud to the final point cloud	
	*final = *scene;

	//
	//  Visualization Module
	//
	if(use_viewer_) {
		if(use_fovea)
			viewer = new pcl::visualization::PCLVisualizer("Multiresolution Correspondence");
		else
			viewer = new pcl::visualization::PCLVisualizer("Correspondence WITHOUT multiresolution");
			
		viewer->initCameraParameters();
		
		if(show_background_white_)
		  viewer->setBackgroundColor (1.0, 1.0, 1.0, 0);	// Setting background to a dark grey

		//viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scene_cloud");
	
	        //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(scene);

		//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> rgb(scene);
		
		viewer->addPointCloud (scene, "scene_cloud");

		viewer->resetCameraViewpoint("scene_cloud");

		//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0f, 0.4f, 0.0f, "sample cloud");

		if (use_fovea)
			foveatedScene->printFoveatedModel(viewer);
	}

	pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

	if(use_viewer_) {
			
		if (show_correspondences_ && show_keypoints_) {
			//  We are translating the model so that it doesn't end in the middle of the scene representation
			pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
			pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
	
			pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
			viewer->addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
		}

		if (show_keypoints_) {
			pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
			viewer->addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
			viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

			pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
			viewer->addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
			viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
		}
	}
	
	for (size_t i = 0; i < rototranslations.size (); ++i) {
		pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
		pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);
		

		std::stringstream ss_cloud;
		ss_cloud << "instance" << i;
		
		if(use_viewer_) {
			pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
			viewer->addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());
		}

		for(int j = 0; j < rotated_model->size(); j++) {
			PointType *p = &rotated_model->at(j);
			uint8_t r = 255, g = 0, b = 0; // Example: Red color
			uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
			p->rgb = *reinterpret_cast<float*>(&rgb);
		}  

		// Append the found object model  to the final point cloud 
		*final += *rotated_model;
	
		if (use_viewer_) {
			if (show_correspondences_) {
				for (size_t j = 0; j < clustered_corrs[i].size (); ++j) {
					std::stringstream ss_line;
					ss_line << "correspondence_line" << i << "_" << j;
					PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
					PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);

					//  We are drawing a line for each pair of clustered correspondences found between the model and the scene
					viewer->addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
				}
			}
		}
	} // end for

	
	if(use_fovea)
		pcl::io::savePCDFileASCII("output_foveated_cloud.pcd", *final);
	else
		pcl::io::savePCDFileASCII("output_cloud.pcd", *final);

	
	if(use_viewer_) {
		while (!viewer->wasStopped ())
    			viewer->spinOnce ();
	}
	return (0);
}
