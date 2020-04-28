#ifndef FOVEATED_CLOUD
#define FOVEATED_CLOUD

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
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
#include <pcl/filters/random_sample.h>
//#include <omp.h>


#define FMAX(A, B) ((A) > (B) ? (A) : (B))
#define FMIN(A, B) ((A) < (B) ? (A) : (B))

enum foveatedMethods {FOVEATED_VOXELGRID, FOVEATED_UNIFORMSAMPLING};
template <class T>
class FoveatedCloud {

	public:
		typename pcl::PointCloud<T> *points;
		float min_res, max_res;
		float *x0, *y0, *z0; 	//starting coordinates of the parallelepiped that involves all points
		float fovea_factor;
		int method;
		std::vector<int> acess_cloud;

		FoveatedCloud() {		  
			points = new pcl::PointCloud<T> ();
			setNumbersFovea(1);
			initCoordParameters(1);
			initMainParameters(2, 0.001, 0.01, 0, FOVEATED_VOXELGRID);
			srand(time(NULL));
		}
		
		FoveatedCloud(int num_fovea) {		  
			points = new pcl::PointCloud<T> ();
			setNumbersFovea(num_fovea);
			initCoordParameters(num_fovea);
			initMainParameters(2, 0.001, 0.01, 0, FOVEATED_VOXELGRID);
			srand(time(NULL));
		}

		void setM(int m) {
			assert(m > 0);
			this->m = m;
			initBoundariesFovea();
		}

		void setNumbersFovea(int num_fovea){
		        assert(num_fovea > 0);			
			this->num_fovea = num_fovea;
		}

		void initMainParameters(int m, float max_res, float min_res, float fovea_factor, foveatedMethods methods){
  			this->m = m;
			this->max_res = max_res;
			this->min_res = min_res;
			this->fovea_factor = fovea_factor;
			this->method = methods;
		}

		void initCoordParameters(int num_fovea){
		        x0 = (float*)malloc(num_fovea*sizeof(float));
			y0 = (float*)malloc(num_fovea*sizeof(float));
			z0 = (float*)malloc(num_fovea*sizeof(float));
		  		  
			fx = (float*)malloc(num_fovea*sizeof(float));
			fy = (float*)malloc(num_fovea*sizeof(float));
			fz = (float*)malloc(num_fovea*sizeof(float));

			Ux = (float*)malloc(num_fovea*sizeof(float));
			Uy = (float*)malloc(num_fovea*sizeof(float));
			Uz = (float*)malloc(num_fovea*sizeof(float));

			Wx = (float*)malloc(num_fovea*sizeof(float));
			Wy = (float*)malloc(num_fovea*sizeof(float));
			Wz = (float*)malloc(num_fovea*sizeof(float));
	
			for(int i = 0; i < num_fovea; i++){
			  *(x0 + i) = *(y0 + i) = *(z0 + i) = 0;
			  *(fx + i) = *(fy + i) = *(fz + i) = 0;
			  *(Ux + i) = *(Uy + i) = *(Uz + i) = 1;
			  *(Wx + i) = *(Wy + i) = *(Wz + i) = 1;
			}

		}

		void initBoundariesFovea(){
		        int limits = num_fovea*(m + 1); 
		        deltax = (float*)malloc(limits*sizeof(float));
			deltay = (float*)malloc(limits*sizeof(float));
			deltaz = (float*)malloc(limits*sizeof(float));

			skx = (float*)malloc(limits*sizeof(float));
			sky = (float*)malloc(limits*sizeof(float));
			skz = (float*)malloc(limits*sizeof(float));
			
			for(int i = 0; i < limits; i++){
			  *(deltax+i) = *(deltay+i) = *(deltaz+i) = 0;
			  *(skx+i) = *(sky+i) = *(skz+i) = 1;
			}
		}

		void setStartCoord(float x, float y, float z) {
			*x0 = x;
			*y0 = y;
			*z0 = z;
		}


		void setModelSize(float ux, float uy, float uz) {
			assert(ux > 0 && uy > 0 && uz > 0 && num_fovea == 1);
			*Ux = ux;
			*Uy = uy;
			*Uz = uz;
		}
		
		void setFoveatedLevelSize(float wx, float wy, float wz) {
			assert(wx > 0 && wy > 0 && wz > 0 && num_fovea == 1);
			*Wx = wx;
			*Wy = wy;
			*Wz = wz;
		}

		void setStartCoord(float x, float y, float z, int i) {
  			assert(num_fovea > i && i >= 0);
		        *(x0 + i) = x;
			*(y0 + i) = y;
			*(z0 + i) = z;
		}


		void setModelSize(float ux, float uy, float uz, int i) {
			assert(ux > 0 && uy > 0 && uz > 0 && num_fovea > i && i >= 0);
			*(Ux + i) = ux;
			*(Uy + i) = uy;
			*(Uz + i) = uz;
		}
		
		void setFoveatedLevelSize(float wx, float wy, float wz, int i) {
			assert(wx > 0 && wy > 0 && wz > 0 && num_fovea > i && i >= 0);
			*(Wx + i) = wx;
			*(Wy + i) = wy;
			*(Wz + i) = wz;
		}

		float calcDelta(int k, float U, float W, float f, int m){
		  //return k*(U - W + 2*f)/(2*m);
		  return FMAX(0, (k*(U - W + 2*f))/(2*m) - fovea_factor);
		}

		float calcBoundFoveatedLevel(int k, float U, float W, int m){
		  return (k*(W - U) + m*U)/m;		  
		}

		float calcBoundFoveatedLevel(int k, float U, float W, int m, float delta){
		  //return (k*(W - U) + m*U)/m;
		  return FMIN(U - delta, (k*(W - U) + m*U)/m + 2*fovea_factor);
		}

		//add to the foveated cloud, the foveatedLevel using the provided cloud
		//parameters
		//cloud: the input point cloud, note that it is not changed
		//foveatedLevel: between 0 and (m-1)
		void addCloudLevel3(typename pcl::PointCloud<T>::Ptr cloud, int foveatedLevel, int fovea) {
			assert(foveatedLevel >= 0 && foveatedLevel <= m && fovea < num_fovea);//validation for parameters fovea
			typename pcl::PointCloud<T>::Ptr res(new pcl::PointCloud<T>());
			
			int k = foveatedLevel; //k is a short name for foveatedLevel
			printf("\tSize scene: %lu; Level Fovea: %d\n", cloud->size(), k);
			int numExtracted = 0;
     
			for(int i = 0; i < cloud->size(); i++) {			       
				T p = cloud->at(i);

				
				lx = checkSpaceXFovea(p.x, fovea, k);
				ly = checkSpaceYFovea(p.y, fovea, k);
				lz = checkSpaceZFovea(p.z, fovea, k);

				cx = scheckSpaceXFovea(p.x, fovea, k);
				cy = scheckSpaceYFovea(p.y, fovea, k);
				cz = scheckSpaceZFovea(p.z, fovea, k);

				if((!lx || !ly || !lz) || (cx && cy && cz)) continue;
				//if((!lx || !ly || !lz)) continue;
				
				lx2 = checkSpaceXFovea(p.x, fovea, k + 1);
				ly2 = checkSpaceYFovea(p.y, fovea, k + 1);
				lz2 = checkSpaceZFovea(p.z, fovea, k + 1);

				cx2 = scheckSpaceXFovea(p.x, fovea, k+1);
				cy2 = scheckSpaceYFovea(p.y, fovea, k+1);
				cz2 = scheckSpaceZFovea(p.z, fovea, k+1);

				if(k == m) {
					res->push_back(p);
					numExtracted++;					
				} else {
				  	if(!(lx2 && ly2 && lz2)) {
						res->push_back(p);
						numExtracted++;
						}
				}
			}
			if(method == FOVEATED_VOXELGRID) {
				pcl::VoxelGrid<T> grid;
				pcl::PointCloud<T> tmp;				
				grid.setInputCloud(res);
				float leaf_x, leaf_y, leaf_z;
				float leaf_size = ((max_res - min_res)*k+min_res*m)/m;
				leaf_z = leaf_y = leaf_x = leaf_size;
				grid.setLeafSize (leaf_x, leaf_y, leaf_z);
				grid.filter(tmp);		 
				*points += tmp;
				printf("\tScene points sampled: %lu; Level Fovea: %d; Fovea Number: %d\n", points->size(), k, fovea);
			} else { //FOVEATED_UNIFORMSAMPLING
				pcl::UniformSampling<T> uniform_sampling;
				pcl::PointCloud<int> sampled_indices;
				pcl::PointCloud<T> tmp;				
				uniform_sampling.setInputCloud(res);
				float radSearch = ((min_res - max_res)*k+max_res*m)/m;
				uniform_sampling.setRadiusSearch(radSearch);
				//uniform_sampling.compute (sampled_indices);
				pcl::copyPointCloud (*res, sampled_indices.points, tmp);
				uniform_sampling.filter (tmp);				
				*points += tmp;				
				printf("\tKeypoints sampled: %lu; Level Fovea: %d; Fovea Number: %d\n", points->size(), k, fovea);
				}
		}


		void clear() {
			points->clear();
		}

		//foveatedLevel: between 0 and (m-1)
		void printFoveatedModel(pcl::visualization::PCLVisualizer *viewer) {
		    for(int i = 0; i < num_fovea; i++){
		      double r = ((double) rand() / (RAND_MAX));
		      double g = ((double) rand() / (RAND_MAX));
		      double b = ((double) rand() / (RAND_MAX));
	
			for(int k = 0; k <= m; k++) {
			  
			        float deltax = *(this->deltax + k + i*(m + 1));
				float deltay = *(this->deltay + k + i*(m + 1));
				float deltaz = *(this->deltaz + k + i*(m + 1));

				float skx = *(this->skx + k + i*(m + 1));
				float sky = *(this->sky + k + i*(m + 1));
				float skz = *(this->skz + k + i*(m + 1));

				
				pcl::PointXYZ p1(*(x0 + i) + deltax,	   *(y0 + i) + deltay,	     *(z0 + i) + deltaz);
				pcl::PointXYZ p2(*(x0 + i) + deltax + skx, *(y0 + i) + deltay,	     *(z0 + i) + deltaz);
				pcl::PointXYZ p3(*(x0 + i) + deltax + skx, *(y0 + i) + deltay+sky,   *(z0 + i) + deltaz);
				pcl::PointXYZ p4(*(x0 + i) + deltax,	   *(y0 + i) + deltay + sky, *(z0 + i) + deltaz);

				pcl::PointXYZ p5(*(x0 + i) + deltax,	   *(y0 + i) + deltay,	        *(z0 + i) + deltaz + skz);
				pcl::PointXYZ p6(*(x0 + i) + deltax + skx, *(y0 + i) + deltay,	        *(z0 + i) + deltaz + skz);
				pcl::PointXYZ p7(*(x0 + i) + deltax + skx, *(y0 + i) + deltay + sky,	*(z0 + i) + deltaz + skz);
				pcl::PointXYZ p8(*(x0 + i) + deltax,	   *(y0 + i) + deltay + sky,	*(z0 + i) + deltaz + skz);

				char linename[500];
				sprintf(linename, "linha1-%d-f%d\n", k, i);
				viewer->addLine(p1, p2, r, g, b, linename);
				sprintf(linename, "linha2-%d-f%d\n", k, i);
				viewer->addLine(p2, p3, r, g, b, linename);
				sprintf(linename, "linha3-%d-f%d\n", k, i);
				viewer->addLine(p3, p4, r, g, b, linename);
				sprintf(linename, "linha4-%d-f%d\n", k, i);
				viewer->addLine(p4, p1, r, g, b, linename);
				
				sprintf(linename, "linha5-%d-f%d\n", k, i);
				viewer->addLine(p5, p6, r, g, b, linename);
				sprintf(linename, "linha6-%d-f%d\n", k, i);
				viewer->addLine(p6, p7, r, g, b, linename);
				sprintf(linename, "linha7-%d-f%d\n", k, i);
				viewer->addLine(p7, p8, r, g, b, linename);
				sprintf(linename, "linha8-%d-f%d\n", k, i);
				viewer->addLine(p8, p5, r, g, b, linename);
				
				sprintf(linename, "linha9-%d-f%d\n", k, i);
				viewer->addLine(p1, p5, r, g, b, linename);
				sprintf(linename, "linha10-%d-f%d\n", k, i);
				viewer->addLine(p2, p6, r, g, b, linename);
				sprintf(linename, "linha11-%d-f%d\n", k, i);
				viewer->addLine(p3, p7, r, g, b, linename);
				sprintf(linename, "linha12-%d-f%d\n", k, i);
				viewer->addLine(p4, p8, r, g, b, linename);
				printf("\tDraw Fovea: %d; Level Fovea: %d\n", i, k);

			}
		    }
		}

	void ensureFoveaLimits() {
		*fx = (*fx < (*Wx - *Ux)/2 ? (*Wx - *Ux)/2 : (*fx > (*Ux - *Wx)/2 ? (*Ux - *Wx)/2 : *fx));
		*fy = (*fy < (*Wy - *Uy)/2 ? (*Wy - *Uy)/2 : (*fy > (*Uy - *Wy)/2 ? (*Uy - *Wy)/2 : *fy));
		*fz = (*fz < (*Wz - *Uz)/2 ? (*Wz - *Uz)/2 : (*fz > (*Uz - *Wz)/2 ? (*Uz - *Wz)/2 : *fz));
	}
	
	void genFoveasBoundaries(){
	  for(int fovea = 0; fovea < num_fovea; fovea++) {
	    for(int k = 0; k <= m; k++) {
	      *(deltax + k + fovea*(m + 1)) = calcDelta(k, *(Ux + fovea),  *(Wx + fovea),  *(fx + fovea), m);
	      *(deltay + k + fovea*(m + 1)) = calcDelta(k, *(Uy + fovea),  *(Wy + fovea),  *(fy + fovea), m);
	      *(deltaz + k + fovea*(m + 1)) = calcDelta(k, *(Uz + fovea),  *(Wz + fovea),  *(fz + fovea), m);
	      *(skx + k + fovea*(m + 1)) = calcBoundFoveatedLevel(k, *(Ux + fovea), *(Wx + fovea), m, *(deltax + k + fovea*(m + 1)));
	      *(sky + k + fovea*(m + 1)) = calcBoundFoveatedLevel(k, *(Uy + fovea), *(Wy + fovea), m, *(deltay + k + fovea*(m + 1)));
	      *(skz + k + fovea*(m + 1)) = calcBoundFoveatedLevel(k, *(Uz + fovea), *(Wz + fovea), m, *(deltaz + k + fovea*(m + 1)));
	    }
	  }	  
	}

	void copyFoveasBoundaries(float *deltax, float *deltay, float *deltaz, float *skx, float *sky, float *skz ){
	  this->deltax = deltax;
	  this->deltay = deltay;
	  this->deltaz = deltaz;
	  this->skx = skx;
	  this->sky = sky;
	  this->skz = skz;
	}

	int scheckSpaceXFovea(float p, int fovea, int k){
	  int validation = 0;
	  for(int i = 0; i < fovea && k <= m; i++)
	    validation = (validation || ((p >= *(x0 + i) + *(deltax + k + i*(m + 1))) && (p <= *(x0 + i) + *(deltax + k  + i*(m + 1)) + *(skx + k +  i*(m + 1))) ) );	  
	  return validation;
	}

	int scheckSpaceYFovea(float p, int fovea, int k){
	  int validation = 0;
	  for(int i = 0; i < fovea && k <= m; i++)
	    validation = (validation || ((p >= *(y0 + i) + *(deltay + k + i*(m + 1))) && (p <= *(y0 + i) + *(deltay + k  + i*(m + 1)) + *(sky + k +  i*(m + 1))) ) );
	  return validation;
	}
	
	int scheckSpaceZFovea(float p, int fovea, int k){
	  int validation = 0;
	  for(int i = 0; i < fovea && k <= m; i++)
	    validation = (validation || ((p >= *(z0 + i) + *(deltaz + k + i*(m + 1))) && (p <= *(z0 + i) + *(deltaz + k  + i*(m + 1)) + *(skz + k +  i*(m + 1))) ) );  
	  return validation;
	}

	
	int checkSpaceXFovea(float p, int fovea, int k){
	   return ((p >= *(x0 + fovea) + *(deltax + k + fovea*(m + 1))) && (p <= *(x0 + fovea) + *(deltax + k  + fovea*(m + 1)) + *(skx + k +  fovea*(m + 1))));	  
	}

	int checkSpaceYFovea(float p, int fovea, int k){
	  return ((p >= *(y0 + fovea) + *(deltay + k + fovea*(m + 1))) && (p <= *(y0 + fovea) + *(deltay + k  + fovea*(m + 1)) + *(sky + k +  fovea*(m + 1))));	  
	}
	
	int checkSpaceZFovea(float p, int fovea, int k){
	  return ((p >= *(z0 + fovea) + *(deltaz + k + fovea*(m + 1))) && (p <= *(z0 + fovea) + *(deltaz + k  + fovea*(m + 1)) + *(skz + k +  fovea*(m + 1))));	  	  
	}
	
	
	void ensureFoveaLimits(int i) {
  	        *(fx + i) = (*(fx + i) < (*(Wx + i) - *(Ux + i))/2 ? (*(Wx + i) - *(Ux + i))/2 : (*(fx + i) > (*(Ux + i) - *(Wx + i))/2 ? (*(Ux + i) - *(Wx + i))/2 : *(fx + i)));
		*(fy + i) = (*(fy + i) < (*(Wy + i) - *(Uy + i))/2 ? (*(Wy + i) - *(Uy + i))/2 : (*(fy + i) > (*(Uy + i) - *(Wy + i))/2 ? (*(Uy + i) - *(Wy + i))/2 : *(fy + i)));
		*(fz + i) = (*(fz + i) < (*(Wz + i) - *(Uz + i))/2 ? (*(Wz + i) - *(Uz + i))/2 : (*(fz + i) > (*(Uz + i) - *(Wz + i))/2 ? (*(Uz + i) - *(Wz + i))/2 : *(fz + i)));
	}

	void addVectorToFovea(float dx, float dy, float dz) {
		*fx += dx;
		*fy += dy;
		*fz += dz;
		ensureFoveaLimits();
	}

	void setFovea(float x, float y, float z) {
		*fx = x - *x0 - *Ux/2;
		*fy = y - *y0 - *Uy/2;
		*fz = z - *z0 - *Uz/2;
		ensureFoveaLimits();
	}

	void setFovea(float x, float y, float z, int i) {
	        *(fx + i) = x - *(x0 + i) - *(Ux + i)/2;
		*(fy + i) = y - *(y0 + i) - *(Uy + i)/2;
		*(fz + i) = z - *(z0 + i) - *(Uz + i)/2;
		ensureFoveaLimits(i);
	}


	float getFoveax() {
		return *fx;
	}
	float getFoveay() {
		return *fy;
	}
	float getFoveaz() {
		return *fz;
	}

	float getFoveax(int i) {
	        return *(fx + i);
	}
	float getFoveay(int i) {
	        return *(fy + i);
	}
	float getFoveaz(int i) {
	        return *(fz + i);
	}

	float  * getDeltax() {
	        return deltax;
	}
	float * getDeltay() {
	        return deltay;
	}
	float * getDeltaz() {
	        return deltaz;
	}

       float  * getSkx() {
	        return skx;
	}
	float * getSky() {
	        return sky;
	}
	float * getSkz() {
	        return skz;
	}



	int getM() {
		return m;
	}

	void copyParameters(FoveatedCloud<T> &dst) {
	        dst.fx = fx;
		dst.fy = fy;
		dst.fz = fz;
		dst.Wx = Wx;
		dst.Wy = Wy;
		dst.Wz = Wz;
		dst.Ux = Ux;
		dst.Uy = Uy;
		dst.Uz = Uz;
		dst.m = m;
		dst.x0 = x0;
		dst.y0 = y0;
		dst.z0 = z0;
		dst.deltax = deltax;
		dst.deltay = deltay;
		dst.deltaz = deltaz;
		dst.skx = skx;
		dst.sky = sky;
		dst.sky = sky;
		dst.fovea_factor = fovea_factor;
	}

	void copyBoundaries(FoveatedCloud<T> &dst) {
		dst.deltax = deltax;
		dst.deltay = deltay;
		dst.deltaz = deltaz;
		dst.skx = skx;
		dst.sky = sky;
		dst.sky = sky;
	}

	private:
		float *fx, *fy, *fz;
		float *Ux, *Uy, *Uz; 	         //size of the parallelepiped
		float *Wx, *Wy, *Wz; 	         //size of the (m-1) foveated level
		int m;			         //number of levels - 1, levels are from 0 (coarsest level) to m-1 (higher resolution level)
		int num_fovea;
		int lx, ly, lz, lx2, ly2, lz2;
		int cx, cy, cz, cx2, cy2, cz2;
		float *deltax, *deltay, *deltaz; //internal limits in the directions of the foveas
		float *skx, *sky, *skz;          //external limits in the directions of the foveas
};


#endif

