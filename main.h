/*
 * main.h
 *
 *  Created on: 02/gen/2016
 *      Author: giorgio
 */

#ifndef MAIN_H_
#define MAIN_H_

#include <stdio.h>
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/config.h>
#include <assimp/mesh.h>

#include <algorithm>
#include "cutil_math.h"
#include <vector>
#include <fstream>

#include <sstream>

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"


// segmentation parameters
#define FX 572.41140
#define FY 573.57043
#define CX 325.26110
#define CY 242.04899

/*
 #define FX 571.9737f
#define FY 571.0073f
#define CX 319.5000f
#define CY 239.5000f
*/
#define tomm_ 1000.0f
#define tocast_ 0.5f
#define tom_ 0.001f
//CHALLENGE 1 DATASET
/*double fx = 571.9737;
double fy = 571.0073;
double cx = 319.5000;
double cy = 239.5000;
*/
//ACCV
double fx=572.41140;
double fy=573.57043;
double cx = 325.26110;
double cy = 242.04899;

float k_vec[9] = {static_cast<float>(fx), 0, static_cast<float>(cx), 0, static_cast<float>(fy), static_cast<float>(cy), 0.f,0.f,1.f};

int k=50;//67;//30; //0.003 /10000
int kx=2000;//2.0f;
int ky=30;//0.03f;
int ks=49;//0.05f;
float kdv=4.5f;
float kdc=0.1f;
float min_size=500.0f;
float sigma=0.8f;
float max_ecc = 0.978f;
float max_L1 = 13800.0f;
float max_L2 = 1950.0f;

int DTH = 30; //[mm]
int plusD = 7; //for depth boundary
int point3D = 5; //10//for contact boundary
int g_angle = 148; //154;//148;//2.f/3.f*M_PI;
int l_angle = 151; //56; //M_PI/3.f;
int Lcanny = 50;
int Hcanny = 75;
int FarObjZ = 1500;//1800; //[mm]
cv::Mat kinect_rgb_img;
cv::Mat kinect_depth_img_mm;



#define strideDimRandGem_ 512
#define Nparticle_ 1024//2048//4096
#define Ndim_ 7
//#define SizeXDim_ Nparticle_*Ndim_
#define dimRandGen_ strideDimRandGem_*Nparticle_
#define tx_ 0
#define ty_ 1
#define tz_ 2
#define q0_ 3
#define q1_ 4
#define q2_ 5
#define q3_ 6
#define MIN_NORM_ 1.0e-7f
#define mTs_ 0.5 //1.5f!!!!!!!
#define w_max_ 0.7298f // max inertia weight value
#define w_min_ 0.3f // min inertia weight value
#define max_steps_ 20
#define max_best_fit_ 1.0e10f

//PSO c1 c2 W
#define c1_ 1.0f
#define c2_ 1.0f
#define psoW_ 0.3


//to perturb the quaternion
/*#define aminX_ -15.f
#define amaxX_ 15.f
#define aminY_ -15.f
#define amaxY_ 15.f
#define aminZ_ -15.f
#define amaxZ_ 15.f
*/
#define aminX_ -10.f
#define amaxX_ 10.f
#define aminY_ -10.f
#define amaxY_ 10.f
#define aminZ_ -10.f
#define amaxZ_ 10.f

//to perturb linear velocity
#define vdot_ 3.f
#define rows_ 480
#define cols_ 640
#define _1cols_ 1.0f/640.0f
#define _2_p32_ 4294967296.0

#define NumStreams 16
#define ForEachParticle_ (int)(((float)Nparticle_)/((float)NumStreams))

#define RxC_ rows_*cols_
#define _2RxC_ 2*RxC_
#define _3RxC_ 3*RxC_
#define RenderedDepthBufferSize_ Nparticle_*RxC_

//#define FX 571.9737f
//#define FY 571.0073f
//#define CX 319.5000f
//#define CY 239.5000f

//#define tomm_ 1000.0f
//#define tom_ 0.001f
//#define tocast_ 0.5f

#define Z0 0.f
#define Z100 0.1f//meters
#define Z100_2 Z100*Z100//meters

#define kEpsilon 1e-8f

#define kInfinity 3.4e38f //std::numeric_limits<float>::max();
#define fxptZero_ 0
#define overA_ ((float)(1<<8))/(256.f)
#define maxVal_depthBuffer 60000.0f
#define maxVal_depthBufferInt 60000
#define minVal_depthBuffer 0.0f
#define max_AABB_ 50000
#define min_AABB_ -1

#define fxptBlocks 6
#define fxptThsPerBlock 512
#define _4fxptBlocks_ fxptBlocks*4
#define _warpSize_ 32
#define _4warpSize_ _warpSize_*4
#define _2warpSize_ _warpSize_*2
#define _8warpSize_ _warpSize_*8

#define _6Nparticle_ 6*Nparticle_
#define _12Nparticle_ 12*Nparticle_
#define _18Nparticle_ 18*Nparticle_


struct __align__(16) OrderedOBJVertices_t
{
    float* x; //4 byte each: 4x4=16byte
    float* y;
    float* z;
};

struct __align__(16) QPSOParams_t
{

    float c1; // cognitive coefficient
    float c2; // social coefficient

    float x_lo[3]; // lower range limit for each
    float x_hi[3];

    unsigned int num_point_SegmentedCluster;

    uint32_t numVerts;
    uint32_t numFaces;

    int AABBminX;
    int AABBmaxX;
    int AABBminY;
    int AABBmaxY;

};


__constant__ QPSOParams_t d_qpso_params[1];

//__constant__  float shampoo1024const[1024*9];

/* DEVICE PSO QUAT FUNCTIONS */
__device__ __forceinline__
void NormalizeQuaternionPSOstateVect(float *q)
{
    const float denom = sqrtf((q[q0_])*(q[q0_]) + (q[q1_])*(q[q1_]) + (q[q2_])*(q[q2_]) + (q[q3_])*(q[q3_]));
    if(denom > MIN_NORM_) {
        const float _1overDenom = __frcp_rn(denom);
        q[q0_] = (q[q0_])*_1overDenom;
        q[q1_] = (q[q1_])*_1overDenom;
        q[q2_] = (q[q2_])*_1overDenom;
        q[q3_] = (q[q3_])*_1overDenom;
    }

}

__device__ __forceinline__
void Cast2TopHalfHyperspherePSOstateVect(float* q)
{
    if(q[q0_]<0.0f)//q0
    {
        q[q0_] = -q[q0_];
        q[q1_] = -q[q1_];
        q[q2_] = -q[q2_];
        q[q3_] = -q[q3_];
    }

}
__device__ __forceinline__
void Cast2TopHalfHypersphere(float* q)
{
    if(q[0]<0.0f)//q0
    {
        q[0] = -q[0];
        q[1] = -q[1];
        q[2] = -q[2];
        q[3] = -q[3];
    }

}

__device__ __forceinline__
void QuatProd_pqPSOstateVect(const float* p, const float* q, float* q_tilde )
{
    q_tilde[0] = p[q0_]*q[q0_] - p[q1_]*q[q1_] - p[q2_]*q[q2_] - p[q3_]*q[q3_];
    q_tilde[1] = p[q1_]*q[q0_] + p[q0_]*q[q1_] + p[q2_]*q[q3_] - p[q3_]*q[q2_];
    q_tilde[2] = p[q2_]*q[q0_] + p[q0_]*q[q2_] + p[q3_]*q[q1_] - p[q1_]*q[q3_];
    q_tilde[3] = p[q3_]*q[q0_] + p[q0_]*q[q3_] + p[q1_]*q[q2_] - p[q2_]*q[q1_];
}
__device__ __forceinline__
void QuatProd_pq_1PSOstateVect(const float* p, const float* q, float* q_tilde )
{
    //q became the coniugate q=(s,vx,vy,vz) -> q"=(s,-vx,-vy,-vz)
    q_tilde[0] = p[q0_]*q[q0_] + p[q1_]*q[q1_] + p[q2_]*q[q2_] + p[q3_]*q[q3_];
    q_tilde[1] = p[q1_]*q[q0_] - p[q0_]*q[q1_] - p[q2_]*q[q3_] + p[q3_]*q[q2_];
    q_tilde[2] = p[q2_]*q[q0_] - p[q0_]*q[q2_] - p[q3_]*q[q1_] + p[q1_]*q[q3_];
    q_tilde[3] = p[q3_]*q[q0_] - p[q0_]*q[q3_] - p[q1_]*q[q2_] + p[q2_]*q[q1_];
}

__device__ __forceinline__
void QuatKinematicsPSOStateVectWithInit(float* qinit, float* q, const float* w)
{
    //q[7] ; w[7]
    /** OverWrite q that is pso_pos[i][q0:q3]**/
    //w[q0_]=0.0 because w=[vx;vy;vz;0;wx;wy;wz]

    float w_temp[7]={0.f};

    w_temp[q1_] = w[q1_]*mTs_;
    w_temp[q2_] = w[q2_]*mTs_;
    w_temp[q3_] = w[q3_]*mTs_;

    const float norm_pso_vel_quat = sqrtf(w_temp[q1_]*w_temp[q1_]+
                                    w_temp[q2_]*w_temp[q2_]+
                                    w_temp[q3_]*w_temp[q3_]);
    float cosW=0.0f;
    float sinW=0.0f;
    //float Tc=1.0;
    const float domega = norm_pso_vel_quat*0.5f;
    cosW = cosf(domega);
    if (domega<MIN_NORM_) {
        sinW=1.0f;
    }
    else{
        const float _1overDomega = __frcp_rn(domega);
        sinW = sinf(domega)*_1overDomega;
    }

    float qXw[4];

    QuatProd_pqPSOstateVect(qinit,w_temp,qXw);

    #pragma unroll
    for (int w_idx=q0_; w_idx<=q3_; ++w_idx) {

        q[w_idx] = cosW*qinit[w_idx] + 0.5f*sinW*qXw[w_idx-q0_];
    }
    NormalizeQuaternionPSOstateVect(q);
    Cast2TopHalfHyperspherePSOstateVect(q);

}
__host__ __device__ __forceinline__
void quat_vect_crossPSOstateVect(const float* quat, const float* vec, float* qXv)
{
    /* qXv = cross(quat.xyz, vec) */

    qXv[0] = quat[q2_]*vec[2]-quat[q3_]*vec[1];
    qXv[1] =-(quat[q1_]*vec[2]-quat[q3_]*vec[0]);
    qXv[2] = quat[q1_]*vec[1]-quat[q2_]*vec[0];

}
__host__ __device__ __forceinline__
void quat_vect_crossPSOstateVect(const float* quat, const float4& vec, float* qXv)
{
    /* qXv = cross(quat.xyz, vec) */

    qXv[0] = quat[q2_]*vec.z-quat[q3_]*vec.y;
    qXv[1] =-(quat[q1_]*vec.z-quat[q3_]*vec.x);
    qXv[2] = quat[q1_]*vec.y-quat[q2_]*vec.x;

}
__host__ __device__ __forceinline__
void quatVectRotationPSOstateVect(const float* quat, const float* vec, float* pvec)
{

    /*  t = 2 * cross(q.xyz, v)
        v' = v + q.w * t + cross(q.xyz, t)
    */
    float t[3];
    float k[3];
    quat_vect_crossPSOstateVect(quat,vec,&t[0]);
    t[0] = 2.0f*t[0];t[1] = 2.0f*t[1];t[2] = 2.0f*t[2];

    quat_vect_crossPSOstateVect(quat,t,&k[0]);

    pvec[0] = vec[0] + quat[q0_]*t[0] + k[0];
    pvec[1] = vec[1] + quat[q0_]*t[1] + k[1];
    pvec[2] = vec[2] + quat[q0_]*t[2] + k[2];



}
__host__ __device__ __forceinline__
void quatVectRotationPSOstateVect(const float* quat, const float4& vec, float* pvec)
{

    /*  t = 2 * cross(q.xyz, v)
        v' = v + q.w * t + cross(q.xyz, t)
    */
    float t[3];
    float k[3];
    quat_vect_crossPSOstateVect(quat,vec,&t[0]);
    t[0] = 2.0f*t[0];t[1] = 2.0f*t[1];t[2] = 2.0f*t[2];

    quat_vect_crossPSOstateVect(quat,t,&k[0]);

    pvec[0] = vec.x + quat[q0_]*t[0] + k[0];
    pvec[1] = vec.y + quat[q0_]*t[1] + k[1];
    pvec[2] = vec.z + quat[q0_]*t[2] + k[2];



}



__device__ __forceinline__
void QuatKinematicsPSOStateVect(float* q, const float* w)
{
    //q[7] ; w[7]
    /** OverWrite q that is pso_pos[i][q0:q3]**/
    //w[q0_]=0.0 because w=[vx;vy;vz;0;wx;wy;wz]

    float w_temp[7]={0.f};

    w_temp[q1_] = w[q1_]*mTs_;
    w_temp[q2_] = w[q2_]*mTs_;
    w_temp[q3_] = w[q3_]*mTs_;

    const float norm_pso_vel_quat = sqrtf(w_temp[q1_]*w_temp[q1_]+
                                    w_temp[q2_]*w_temp[q2_]+
                                    w_temp[q3_]*w_temp[q3_]);
    float cosW=0.0f;
    float sinW=0.0f;
    //double Tc=5;
    const float domega = norm_pso_vel_quat*0.5f;
    cosW = cosf(domega);
    if (domega<MIN_NORM_) {
        sinW=1.0f;
    }
    else{
        const float _1overDomega = __frcp_rn(domega);
        sinW = sinf(domega)*_1overDomega;
    }

    float qXw[4];
    //q[7], w[7], qxw[4]
    QuatProd_pqPSOstateVect(q,w_temp,qXw);

    #pragma unroll
    for (int w_idx=q0_; w_idx<=q3_; ++w_idx) {

        q[w_idx] = cosW*q[w_idx] + 0.5f*sinW*qXw[w_idx-q0_];
    }
    NormalizeQuaternionPSOstateVect(q);
    Cast2TopHalfHyperspherePSOstateVect(q);
}

__device__ __forceinline__
void projectModelPointsToPixels(const float* pso_pose_vec, const float* point, uint16_t* UVz)
{
    /* input:
     * @pso_pose_vec: is the particle state vector:
     *                pose[tx,ty,tz,q0,q1,q2,q3] in [meter, unit_quat]
     * @point: is a 3D point [in meter]
     * output:
     * @UVz: is the normalized pixel coordinate UV of
     *       the projected point and its depth value Z in millimeters.
     */
    /* UVZ = K*T*XYZ; UV = UVZ/Z; */
    //Rotation
    float pprime[3];
    quatVectRotationPSOstateVect(pso_pose_vec,point,&pprime[0]);
    //Translation
    pprime[tx_] += pso_pose_vec[tx_];
    pprime[ty_] += pso_pose_vec[ty_];
    pprime[tz_] += pso_pose_vec[tz_];
    //Projection K and normalize
    /*  |cx*z + fx*x    |       |cx + fx*(x/z)|
        |cy*z + fy*y|  ==>  |cy + fy*(y/z)|
        |z          |       |     1       |
     */

    UVz[tx_] = static_cast<uint16_t>( CX + FX*(pprime[tx_]/pprime[tz_]) + tocast_ );
    UVz[ty_] = static_cast<uint16_t>( CY + FY*(pprime[ty_]/pprime[tz_]) + tocast_ );
    UVz[tz_] = static_cast<uint16_t>(tomm_*pprime[tz_] + tocast_);//in mm

    return;
}

__device__ __forceinline__
void projectModelPointsToPixels(const float* pso_pose_vec, const float* point, float4* UVz)
{
    /* input:
     * @pso_pose_vec: is the particle state vector:
     *                pose[tx,ty,tz,q0,q1,q2,q3] in [meter, unit_quat]
     * @point: is a 3D point [in meter]
     * output:
     * @UVz: is the normalized pixel coordinate UV of
     *       the projected point and its depth value Z in millimeters (but all in float).
     */
    /* UVZ = K*T*XYZ; UV = UVZ/Z; */
    //Rotation
    float pprime[3];
    quatVectRotationPSOstateVect(pso_pose_vec,point,&pprime[0]);
    //Translation
    pprime[tx_] += pso_pose_vec[tx_];
    pprime[ty_] += pso_pose_vec[ty_];
    pprime[tz_] += pso_pose_vec[tz_];
    //Projection K and normalize
    /*  |cx*z + fx*x    |       |cx + fx*(x/z)|
        |cy*z + fy*y|  ==>  |cy + fy*(y/z)|
        |z          |       |     1       |
     */

    //USING MACROs we gain 20 ms over __constant__
    UVz->x = ( CX + FX*(pprime[tx_]/pprime[tz_]) + tocast_ );
    UVz->y = ( CY + FY*(pprime[ty_]/pprime[tz_]) + tocast_ );
    UVz->z = (tomm_*pprime[tz_] + tocast_);//in mm
    UVz->w = 0.f;
    return;
}


__device__ __forceinline__
void projectModelPointsToPixels(const float* pso_pose_vec, const float4& point, float4* UVz)
{
    /* input:
     * @pso_pose_vec: is the particle state vector:
     *                pose[tx,ty,tz,q0,q1,q2,q3] in [meter, unit_quat]
     * @point: is a 3D point [in meter] of type float4 with .w not considered
     * output:
     * @UVz: is the normalized pixel coordinate UV of
     *       the projected point and its depth value Z in millimeters (but all in float).
     */
    /* UVZ = K*T*XYZ; UV = UVZ/Z; */
    //Rotation
    float pprime[3];
    quatVectRotationPSOstateVect(pso_pose_vec,point,&pprime[0]);
    //Translation
    pprime[tx_] += pso_pose_vec[tx_];
    pprime[ty_] += pso_pose_vec[ty_];
    pprime[tz_] += pso_pose_vec[tz_];
    //Projection K and normalize
    /*  |cx*z + fx*x    |       |cx + fx*(x/z)|
        |cy*z + fy*y|  ==>  |cy + fy*(y/z)|
        |z          |       |     1       |
     */

    //USING MACROs we gain 20 ms over __constant__
    UVz->x = ( CX + FX*(pprime[tx_]/pprime[tz_]) + tocast_ );
    UVz->y = ( CY + FY*(pprime[ty_]/pprime[tz_]) + tocast_ );
    UVz->z = (tomm_*pprime[tz_] + tocast_);//in mm
    UVz->w = 0.f;
    return;
}
__device__ __forceinline__
void projectModelPointsToPixels(const float* pso_pose_vec, const float4& point, float4& UVz)
{
    /* input:
     * @pso_pose_vec: is the particle state vector:
     *                pose[tx,ty,tz,q0,q1,q2,q3] in [meter, unit_quat]
     * @point: is a 3D point [in meter] of type float4 with .w not considered
     * output:
     * @UVz: is the normalized pixel coordinate UV of
     *       the projected point and its depth value Z in millimeters (but all in float).
     */
    /* UVZ = K*T*XYZ; UV = UVZ/Z; */
    //Rotation
    float pprime[3];
    quatVectRotationPSOstateVect(pso_pose_vec,point,&pprime[0]);
    //Translation
    pprime[tx_] += pso_pose_vec[tx_];
    pprime[ty_] += pso_pose_vec[ty_];
    pprime[tz_] += pso_pose_vec[tz_];
    //Projection K and normalize
    /*  |cx*z + fx*x    |       |cx + fx*(x/z)|
        |cy*z + fy*y|  ==>  |cy + fy*(y/z)|
        |z          |       |     1       |
     */

    //USING MACROs we gain 20 ms over __constant__
    const float _1pprimeZ = __frcp_rn(pprime[tz_]);
    UVz.x = ( CX + FX*(pprime[tx_]*_1pprimeZ)); //+ tocast_ );
    UVz.y = ( CY + FY*(pprime[ty_]*_1pprimeZ)); //+ tocast_ );
    UVz.z = (tomm_*pprime[tz_]); //+ tocast_);//in mm
    UVz.w = 0.f;
    return;
}

__forceinline__ __host__ void loadOBJ2(const std::string& path_to_obj, OrderedOBJVertices_t& ord_obj, int& numVerts,
                    int& numFaces)
{
    const aiScene *scene = aiImportFile(path_to_obj.c_str(),aiProcessPreset_TargetRealtime_Fast);//aiProcessPreset_TargetRealtime_Fast has the configs you'll need

    aiMesh *mesh = scene->mMeshes[0]; //assuming you only want the first mesh
    //    float *vertexArray;

    //int numVerts;
    numVerts = mesh->mNumFaces*3;
    numFaces = mesh->mNumFaces;
    //OrderedOBJVertices_t ord_obj;
    ord_obj.x = (float*)calloc(numVerts,sizeof(float));
    ord_obj.y = (float*)calloc(numVerts,sizeof(float));
    ord_obj.z = (float*)calloc(numVerts,sizeof(float));

    printf("NumVerts: %d\n",numVerts);
    printf("NumFaces: %d\n",numFaces);
    //    vertexArray = new float[mesh->mNumFaces*3*3];
    for(unsigned int i=0;i<mesh->mNumFaces;i++)
    {
        const aiFace& face = mesh->mFaces[i];

        for(int j=0;j<3;j++)
        {

            aiVector3D pos = mesh->mVertices[face.mIndices[j]];
            ord_obj.x[i*3+j] = pos.x;
            ord_obj.y[i*3+j] = pos.y;
            ord_obj.z[i*3+j] = pos.z;
        }
    }
    //END LOAD OBJ
}

__forceinline__ __host__ void loadOBJ2(const std::string& path_to_obj, std::vector<float>& ord_obj_vect, int& numVerts,
                    int& numFaces)
{
    const aiScene *scene = aiImportFile(path_to_obj.c_str(),aiProcessPreset_TargetRealtime_Fast);//aiProcessPreset_TargetRealtime_Fast has the configs you'll need

    aiMesh *mesh = scene->mMeshes[0]; //assuming you only want the first mesh
    //    float *vertexArray;

    //int numVerts;
    numVerts = mesh->mNumFaces*3;
    numFaces = mesh->mNumFaces;

    printf("NumVerts: %d\n",numVerts);
    printf("NumFaces: %d\n",numFaces);

    std::vector<float> x_vec, y_vec, z_vec;
    for(unsigned int i=0;i<mesh->mNumFaces;++i)
    {
        const aiFace& face = mesh->mFaces[i];

        for(int j=0;j<3;j++)
        {

            aiVector3D pos = mesh->mVertices[face.mIndices[j]];
            x_vec.push_back(pos.x);
            y_vec.push_back(pos.y);
            z_vec.push_back(pos.z);
        }
    }
    ord_obj_vect.clear();
    ord_obj_vect.reserve(3*numVerts);
    ord_obj_vect.insert(ord_obj_vect.end(),x_vec.begin(),x_vec.end());
    ord_obj_vect.insert(ord_obj_vect.end(),y_vec.begin(),y_vec.end());
    ord_obj_vect.insert(ord_obj_vect.end(),z_vec.begin(),z_vec.end());


}

__forceinline__ __host__ void loadOBJ3(const std::string& path_to_obj, std::vector<float>& ord_obj_vect, int& numVerts,
                    int& numFaces)
{
    /*
     * we assumes one thread per triangle (Face) and we store
     * the obj_model as [x0f0 x0f1 x0f2...x0fN |
     *                  x1f0 x1f1 x1f2...x1fN  |
     *                  x2f0 x2f1 x2f2...x2fN  | numVerts
     *                  y0f0 y0f1 y0f2...y0fN  |
     *                  y1f0 y1f1 y1f2...y1fN  |
     *                  y2f0 y2f1 y2f2...y2fN  | 2*numVerts
     *                  z0f0 z0f1 z0f2...z0fN  |
     *                  z1f0 z1f1 z1f2...z1fN  |
     *                  z2f0 z2f1 z2f2...z2fN  ] 3*numVerts
     *
     * where x0 y0 z0 is the first vertex of the face 0 (f0)...or face 58 (f58)
     */


    const aiScene *scene = aiImportFile(path_to_obj.c_str(),aiProcessPreset_TargetRealtime_Fast);//aiProcessPreset_TargetRealtime_Fast has the configs you'll need

    aiMesh *mesh = scene->mMeshes[0]; //assuming you only want the first mesh
    //    float *vertexArray;

    //int numVerts;
    numVerts = mesh->mNumFaces*3;
    numFaces = mesh->mNumFaces;

    printf("NumVerts: %d\n",numVerts);
    printf("NumFaces: %d\n",numFaces);

    std::vector<float> x_vec, y_vec, z_vec;
    ord_obj_vect.clear();
    ord_obj_vect.resize(3*numVerts);
    //simulate each GPU thread per face
    for(unsigned int i=0;i<mesh->mNumFaces;++i)
    {
        const aiFace& face = mesh->mFaces[i];

        for(int j=0;j<3;j++)//three vertices of the face
        {

            aiVector3D pos = mesh->mVertices[face.mIndices[j]];
            ord_obj_vect[i + 0 + j*numFaces] = pos.x;
            ord_obj_vect[i + 3*numFaces + j*numFaces] = pos.y;
            ord_obj_vect[i + 6*numFaces + j*numFaces] = pos.z;

//            x_vec.push_back(pos.x);
//            y_vec.push_back(pos.y);
//            z_vec.push_back(pos.z);
        }
    }
//    ord_obj_vect.clear();
//    ord_obj_vect.reserve(3*numVerts);
//    ord_obj_vect.insert(ord_obj_vect.end(),x_vec.begin(),x_vec.end());
//    ord_obj_vect.insert(ord_obj_vect.end(),y_vec.begin(),y_vec.end());
//    ord_obj_vect.insert(ord_obj_vect.end(),z_vec.begin(),z_vec.end());


}

__forceinline__ __host__ void loadOBJ4(const std::string& path_to_obj, std::vector<float4>& ord_obj_vect, int& numVerts,
                    int& numFaces)
{
    /* USED TO FILL THE TEXTURE MEMORY
     * vector<float4> [v00 v10 v20, v01,v11,v21, ...v0nFaces,v1nFaces,v2nFaces] where vij is a vertex i (float4) of the face j
     */


    const aiScene *scene = aiImportFile(path_to_obj.c_str(),aiProcessPreset_TargetRealtime_Fast);//aiProcessPreset_TargetRealtime_Fast has the configs you'll need

    aiMesh *mesh = scene->mMeshes[0]; //assuming you only want the first mesh
    //    float *vertexArray;

    //int numVerts;
    numVerts = mesh->mNumFaces*3;
    numFaces = mesh->mNumFaces;

    printf("NumVerts: %d\n",numVerts);
    printf("NumFaces: %d\n",numFaces);

    ord_obj_vect.clear();
    ord_obj_vect.resize(numVerts);
    //simulate each GPU thread per face
    for(unsigned int i=0;i<mesh->mNumFaces;++i)
    {
        const aiFace& face = mesh->mFaces[i];

        for(int j=0;j<3;j++)//three vertices of the face
        {

            aiVector3D pos = mesh->mVertices[face.mIndices[j]];
//            ord_obj_vect[i + 0 + j*numFaces] = pos.x;
//            ord_obj_vect[i + 3*numFaces + j*numFaces] = pos.y;
//            ord_obj_vect[i + 6*numFaces + j*numFaces] = pos.z;
              ord_obj_vect[3*i + j].x = pos.x;
              ord_obj_vect[3*i + j].y = pos.y;
              ord_obj_vect[3*i + j].z = pos.z;
//            x_vec.push_back(pos.x);
//            y_vec.push_back(pos.y);
//            z_vec.push_back(pos.z);
        }
    }
//    ord_obj_vect.clear();
//    ord_obj_vect.reserve(3*numVerts);
//    ord_obj_vect.insert(ord_obj_vect.end(),x_vec.begin(),x_vec.end());
//    ord_obj_vect.insert(ord_obj_vect.end(),y_vec.begin(),y_vec.end());
//    ord_obj_vect.insert(ord_obj_vect.end(),z_vec.begin(),z_vec.end());


}
__forceinline__ __host__ void loadOBJ5(const std::string& path_to_obj, std::vector<float4>& ord_obj_vect, int& numVerts,
                    int& numFaces)
{
    /* USED TO FILL THE TEXTURE MEMORY
     * vector<float4> [v0f0 v0f1 v0f2, v0f3,v0f4,v0f5, ... v0fNFaces,v0fNFaces,v0fNFaces
     *                 v1f0,v1f1,v1f2  v1f3,v1f4,v1f5, ... v1fNFaces,v1fNFaces,v1fNFaces
     *                 v2f0,v2f1,v2f2  v2f3,v2f4,v2f5, ... v2fNFaces,v2fNFaces,v2fNFaces  ]
     *                 where vij is a vertex i (float4 x,y,z,0) of the face j
     */


    const aiScene *scene = aiImportFile(path_to_obj.c_str(),aiProcessPreset_TargetRealtime_Fast);//aiProcessPreset_TargetRealtime_Fast has the configs you'll need

    aiMesh *mesh = scene->mMeshes[0]; //assuming you only want the first mesh
    //    float *vertexArray;

    //int numVerts;
    numVerts = mesh->mNumFaces*3;
    numFaces = mesh->mNumFaces;

    printf("NumVerts: %d\n",numVerts);
    printf("NumFaces: %d\n",numFaces);

    ord_obj_vect.clear();
    ord_obj_vect.resize(numVerts);
    //simulate each GPU thread per face
    for(int j=0;j<3;j++)//three vertices of the face
    {
        for(unsigned int i=0;i<mesh->mNumFaces;++i)
        {
            const aiFace& face = mesh->mFaces[i];



                aiVector3D pos = mesh->mVertices[face.mIndices[j]];
        //            ord_obj_vect[i + 0 + j*numFaces] = pos.x;
        //            ord_obj_vect[i + 3*numFaces + j*numFaces] = pos.y;
        //            ord_obj_vect[i + 6*numFaces + j*numFaces] = pos.z;
                  ord_obj_vect[numFaces*j + i].x = pos.x;
                  ord_obj_vect[numFaces*j + i].y = pos.y;
                  ord_obj_vect[numFaces*j + i].z = pos.z;
                  ord_obj_vect[numFaces*j + i].w = 0.f;
        //            x_vec.push_back(pos.x);
        //            y_vec.push_back(pos.y);
        //            z_vec.push_back(pos.z);

        }
    }
//    ord_obj_vect.clear();
//    ord_obj_vect.reserve(3*numVerts);
//    ord_obj_vect.insert(ord_obj_vect.end(),x_vec.begin(),x_vec.end());
//    ord_obj_vect.insert(ord_obj_vect.end(),y_vec.begin(),y_vec.end());
//    ord_obj_vect.insert(ord_obj_vect.end(),z_vec.begin(),z_vec.end());


}

__forceinline__ __device__ void modelVertex2Camera(const float* pso_pose_vec,const float* point, float3& ppoint )
{
    //Rotation
    float pprime[3];
    quatVectRotationPSOstateVect(pso_pose_vec,point,&pprime[0]);
    //Translation
    ppoint.x = pprime[tx_] + pso_pose_vec[tx_];
    ppoint.y = pprime[ty_] + pso_pose_vec[ty_];
    ppoint.z = pprime[tz_] + pso_pose_vec[tz_];
}

__forceinline__ __device__ int rayTriangleIntersect(const float3& ray, const float3& vertexCamera0,const float3& vertexCamera1,const float3& vertexCamera2, float& t)
{


    const float3 v0v1 = vertexCamera1 - vertexCamera0;
    const float3 v0v2 = vertexCamera2 - vertexCamera0;
    const float3 pvec = cross(ray,v0v2);
    const float det = dot(v0v1,pvec);

    /*CULLING*/
    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    if (det < kEpsilon) return 0;//false;

    /*NO CULLING*/
    // ray and triangle are parallel if det is close to 0
    //if (fabs(det) < kEpsilon) return 0;

    float invDet = __frcp_rn(det);//1.f / det;

    //float3 tvec = orig - vertexCamera0;
    const float3 tvec= make_float3(-vertexCamera0.x,-vertexCamera0.y,-vertexCamera0.z);
    const float u = dot(tvec,pvec) * invDet;
    if (u < 0 || u > 1) return 0;//false;

    const float3 qvec = cross(tvec,v0v1);
    const float v = dot(ray,qvec) * invDet;
    if (v < 0 || u + v > 1) return 0;//false;

    t = dot(v0v2,qvec) * invDet;

    return 1;//true;

}

__forceinline__ __device__ void renderFixedPoint(const float4& v1,
                                        const float4& v2,
                                        const float4& v3,
                                        int* d_depth_buffer,
                                        const uint32_t& offset_buffer,
                                        const uint32_t& tIdx)
{

    // 28.4 fixed-point coordinates: 28 integer and 4 after the point
    //Scaling factor == 2^4 = 16
    //*ROUND* cast since +0.5f is already added, after the rototranslation, to UV vertices
//  const int Y1 = int(16.0f * v1.y);
//  const int Y2 = int(16.0f * v2.y);
//  const int Y3 = int(16.0f * v3.y);
//
//  const int X1 = int(16.0f * v1.x);
//  const int X2 = int(16.0f * v2.x);
//  const int X3 = int(16.0f * v3.x);


    //+0.5f is NOT added, after the rototranslation, to UV vertices
    const int Y1 = __float2int_rn(16.0f * v1.y);
    const int Y2 = __float2int_rn(16.0f * v2.y);
    const int Y3 = __float2int_rn(16.0f * v3.y);

    const int X1 = __float2int_rn(16.0f * v1.x);
    const int X2 = __float2int_rn(16.0f * v2.x);
    const int X3 = __float2int_rn(16.0f * v3.x);

    // Deltas
    //Scaling factor == 2^4 = 16
    //B
    const int DX12 = X1 - X2;
    const int DX23 = X2 - X3;
    const int DX31 = X3 - X1;
    //A
    const int DY12 = Y1 - Y2;
    const int DY23 = Y2 - Y3;
    const int DY31 = Y3 - Y1;

    //Area with scaling factor 16*16=2^4*2^4 == 2^8
    int triArea = /*std::abs*/(DX12 * DY31 - DX31 * DY12);
    if(triArea<=0)
    {
        return;
    }
    //float _1triArea = overA_/__int2float_rn(triArea);
    //_1triArea /= 256.f;

    // Bounding rectangle
    //Scaling factor 1 = 2^0
    const int minx = (min(min(X1, X2), X3) + 0xF) >> 4;
    const int maxx = (max(max(X1, X2), X3) + 0xF) >> 4;
    const int miny = (min(min(Y1, Y2), Y3) + 0xF) >> 4;
    const int maxy = (max(max(Y1, Y2), Y3) + 0xF) >> 4;

    //AABB zero Area
    if((maxx-minx)<=0 || (maxy-miny)<=0)
        return;

    const float _1triArea = overA_*__frcp_rn(__int2float_rn(triArea));//__fdividef(overA_,__int2float_rn(triArea));

//  printf("triArea: %d\n",triArea);
//  printf("__int2float_rn(triArea): %f\n",__int2float_rn(triArea));
//  printf("__frcp_rn(__int_as_float(triArea)): %f\n",__frcp_rn(__int_as_float(triArea)));
//  printf("_1triArea: %f\n",_1triArea);
    // Fixed-point deltas
    // Scaling factor == 2^8 = 2^4*2^4 = 256
    //B
    const int FDX12 = DX12 << 4;
    const int FDX23 = DX23 << 4;
    const int FDX31 = DX31 << 4;
    //A
    const int FDY12 = DY12 << 4;
    const int FDY23 = DY23 << 4;
    const int FDY31 = DY31 << 4;

    // Half-edge constants
    //Scaling Factor 16*16=2^4*2^4 == 2^8
    int C1 = DY12 * X1 - DX12 * Y1;
    int C2 = DY23 * X2 - DX23 * Y2;
    int C3 = DY31 * X3 - DX31 * Y3;
    // Correct for fill convention
    //Only top-left edges are considered inside the triangle
    //so we can leave the test if(CX1 > 0 && CX2 > 0 && CX3 > 0) strictly greater than 0 without drawing non top-left edges.
    //here +1 is in Scaling factor 2^8 so real is +1 >> 8
    //add +1 without if
    C1 += (DY12 < 0 || (DY12 == 0 && DX12 > 0));
    C2 += (DY23 < 0 || (DY23 == 0 && DX23 > 0));
    C3 += (DY31 < 0 || (DY31 == 0 && DX31 > 0));

    //Scaling factor 2^4
    const int MINY = (miny << 4);
    const int MINX = (minx << 4);
    //Scaling factor 2^8
    int CY1 = C1 + DX12 * MINY - DY12 * MINX;
    int CY2 = C2 + DX23 * MINY - DY23 * MINX;
    int CY3 = C3 + DX31 * MINY - DY31 * MINX;

    //Scaling Factor 2^8
    //float z1 = v1.z;
    const float z2 = (v2.z - v1.z)* _1triArea ;
    const float z3 = (v3.z - v1.z)* _1triArea ;

    for(int y = miny; y < maxy; ++y)
    {
        //Scaling factor 2^8
        int CX1 = CY1;
        int CX2 = CY2;
        int CX3 = CY3;

        for(int x = minx; x < maxx; ++x)
        {

            //Test Pixel inside triangle
            //const int32_t cx1 = (CX1+ 0xF)>>8;
            const int32_t cx2 = (CX2 + 0xF)>>8;
            const int32_t cx3 = (CX3 + 0xF)>>8;

            //mask == 1 if >0; 0 otherwise
            int32_t mask =( ( !((CX1 & 0x80000000) | !CX1)) & ( !((CX2 & 0x80000000) | !CX2)) & ( !((CX3 & 0x80000000) | !CX3))) > 0;

            const int idx = (y*cols_ + x) + offset_buffer;

            float depth = v1.z;
            depth += cx2*z2;
            depth += cx3*z3;

            int depth_int = __float2int_rn(depth);
            //float depth = v1.z + cx2*z2 + cx3*z3;
//          float depth = __fmaf_rn(__int2float_rn(cx2),z2,v1.z);
//          depth = __fmaf_rn(__int2float_rn(cx3),z3,depth);

            //float previousDepthValue = d_depth_buffer[idx];
            int previousDepthValue = d_depth_buffer[idx];

            /* Up to now, it costs lesser a DepthBuffer writing branch than a DepthBuffer writing mask.
             * The Time of a single kernel call drop down to 0.055ms wrt to 0.07ms as before, but
             * the time of 1024 streamed call is left almost unchanged ~17.09-17.19 ms.
             * Using 6 blocks per 512 ths we get improvement of ~0.6ms wrt 12 blks per 256 ths
             */
//          float maxdepth = min(depth, previousDepthValue);
//          depth = (mask & 0x80000000) ? previousDepthValue : maxdepth;
//          d_depth_buffer[idx] = depth;

            /*TODO: Solve memory coalesced problem and
             * the Race condition one to speed things up !
             */
            if(!(mask & 0x80000000) && depth_int<previousDepthValue)
            {
                //printf("depth: %f\n",depth);
                d_depth_buffer[idx] = depth_int;
            }

            //Scaling factor 2^8
            CX1 -= FDY12;
            CX2 -= FDY23;
            CX3 -= FDY31;
        }
        //Scaling factor 2^8
        CY1 += FDX12;
        CY2 += FDX23;
        CY3 += FDX31;

    }

    return;
}

__forceinline__ __device__ void renderFixedPointAllParticle(const float4& v1,
                                        const float4& v2,
                                        const float4& v3,
                                        float* d_depth_buffer,
                                        const uint32_t& tIdx)
{

    //28.4 fixed-point coordinates: 28 integer and 4 after the point
    //Scaling factor == 2^4 = 16
    //*ROUND* cast since +0.5f is already added, after the rototranslation, to UV vertices
//  const int Y1 = int(16.0f * v1.y);
//  const int Y2 = int(16.0f * v2.y);
//  const int Y3 = int(16.0f * v3.y);
//
//  const int X1 = int(16.0f * v1.x);
//  const int X2 = int(16.0f * v2.x);
//  const int X3 = int(16.0f * v3.x);


    //+0.5f is NOT added, after the rototranslation, to UV vertices
    const int Y1 = __float2int_rn(16.0f * v1.y);
    const int Y2 = __float2int_rn(16.0f * v2.y);
    const int Y3 = __float2int_rn(16.0f * v3.y);

    const int X1 = __float2int_rn(16.0f * v1.x);
    const int X2 = __float2int_rn(16.0f * v2.x);
    const int X3 = __float2int_rn(16.0f * v3.x);

    // Deltas
    //Scaling factor == 2^4 = 16
    //B
    const int DX12 = X1 - X2;
    const int DX23 = X2 - X3;
    const int DX31 = X3 - X1;
    //A
    const int DY12 = Y1 - Y2;
    const int DY23 = Y2 - Y3;
    const int DY31 = Y3 - Y1;

    //Area with scaling factor 16*16=2^4*2^4 == 2^8
    int triArea = /*std::abs*/(DX12 * DY31 - DX31 * DY12);
    if(triArea<=0)
    {
        return;
    }
    //float _1triArea = overA_/__int2float_rn(triArea);
    //_1triArea /= 256.f;

    // Bounding rectangle
    //Scaling factor 1 = 2^0
    const int minx = (min(min(X1, X2), X3) + 0xF) >> 4;
    const int maxx = (max(max(X1, X2), X3) + 0xF) >> 4;
    const int miny = (min(min(Y1, Y2), Y3) + 0xF) >> 4;
    const int maxy = (max(max(Y1, Y2), Y3) + 0xF) >> 4;

    //AABB zero Area
    if((maxx-minx)<=0 || (maxy-miny)<=0)
        return;

    const float _1triArea = overA_*__frcp_rn(__int2float_rn(triArea));//__fdividef(overA_,__int2float_rn(triArea));

//  printf("triArea: %d\n",triArea);
//  printf("__int2float_rn(triArea): %f\n",__int2float_rn(triArea));
//  printf("__frcp_rn(__int_as_float(triArea)): %f\n",__frcp_rn(__int_as_float(triArea)));
//  printf("_1triArea: %f\n",_1triArea);
    // Fixed-point deltas
    // Scaling factor == 2^8 = 2^4*2^4 = 256
    //B
    const int FDX12 = DX12 << 4;
    const int FDX23 = DX23 << 4;
    const int FDX31 = DX31 << 4;
    //A
    const int FDY12 = DY12 << 4;
    const int FDY23 = DY23 << 4;
    const int FDY31 = DY31 << 4;

    // Half-edge constants
    //Scaling Factor 16*16=2^4*2^4 == 2^8
    int C1 = DY12 * X1 - DX12 * Y1;
    int C2 = DY23 * X2 - DX23 * Y2;
    int C3 = DY31 * X3 - DX31 * Y3;
    // Correct for fill convention
    //Only top-left edges are considered inside the triangle
    //so we can leave the test if(CX1 > 0 && CX2 > 0 && CX3 > 0) strictly greater than 0 without drawing non top-left edges.
    //here +1 is in Scaling factor 2^8 so real is +1 >> 8
    //add +1 without if
    C1 += (DY12 < 0 || (DY12 == 0 && DX12 > 0));
    C2 += (DY23 < 0 || (DY23 == 0 && DX23 > 0));
    C3 += (DY31 < 0 || (DY31 == 0 && DX31 > 0));

    //Scaling factor 2^4
    const int MINY = (miny << 4);
    const int MINX = (minx << 4);
    //Scaling factor 2^8
    int CY1 = C1 + DX12 * MINY - DY12 * MINX;
    int CY2 = C2 + DX23 * MINY - DY23 * MINX;
    int CY3 = C3 + DX31 * MINY - DY31 * MINX;

    //Scaling Factor 2^8
    //float z1 = v1.z;
    const float z2 = (v2.z - v1.z)* _1triArea ;
    const float z3 = (v3.z - v1.z)* _1triArea ;

    for(int y = miny; y < maxy; ++y)
    {
        //Scaling factor 2^8
        int CX1 = CY1;
        int CX2 = CY2;
        int CX3 = CY3;

        for(int x = minx; x < maxx; ++x)
        {

            //Test Pixel inside triangle
            //const int32_t cx1 = (CX1+ 0xF)>>8;
            const int32_t cx2 = (CX2 + 0xF)>>8;
            const int32_t cx3 = (CX3 + 0xF)>>8;

            //mask == 1 if >0; 0 otherwise
            int32_t mask =( ( !((CX1 & 0x80000000) | !CX1)) & ( !((CX2 & 0x80000000) | !CX2)) & ( !((CX3 & 0x80000000) | !CX3))) > 0;

            const int idx = (y*cols_ + x)*Nparticle_ + tIdx;//(y*cols_ + x) + offset_buffer;

//          float depth = v1.z;
            const float d1 = cx2*z2;
            const float d2 = cx3*z3;
            float depth = v1.z + d1 + d2;


            //float depth = v1.z + cx2*z2 + cx3*z3;
//          float depth = __fmaf_rn(__int2float_rn(cx2),z2,v1.z);
//          depth = __fmaf_rn(__int2float_rn(cx3),z3,depth);

            float previousDepthValue = d_depth_buffer[idx];

            /* Up to now, it costs lesser a DepthBuffer writing branch than a DepthBuffer writing mask.
             * The Time of a single kernel call drop down to 0.055ms wrt to 0.07ms as before, but
             * the time of 1024 streamed call is left almost unchanged ~17.09-17.19 ms.
             * Using 6 blocks per 512 ths we get improvement of ~0.6ms wrt 12 blks per 256 ths
             */
            //FOR ALL Particle Technique gained 0.2ms without if branch buffer
//          float maxdepth = min(depth, previousDepthValue);
//          depth = (mask & 0x80000000) ? previousDepthValue : maxdepth;
//          d_depth_buffer[idx] = depth;

            /*TODO: Solve memory coalesced problem and
             * the Race condition one to speed things up !
             */
            if(!(mask & 0x80000000) && depth<previousDepthValue)
            {
                //printf("depth: %f\n",depth);
                d_depth_buffer[idx] = depth;
            }

            //Scaling factor 2^8
            CX1 -= FDY12;
            CX2 -= FDY23;
            CX3 -= FDY31;
        }
        //Scaling factor 2^8
        CY1 += FDX12;
        CY2 += FDX23;
        CY3 += FDX31;

    }

    return;
}

__forceinline__ __host__ void testAsyncStreamAvailable()
{
    cudaDeviceProp prop;
    int whichDevice;
    HANDLE_ERROR( cudaGetDevice( &whichDevice ) );
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice ) );
    if (!prop.deviceOverlap) {
            printf( "WARNING: Device will not handle overlaps, so no "
                    "speed up from streams\n" );
    }
    else
    {
        printf( "OK for streams !!!\n" );
    }
    return;
}


__forceinline__ __device__ void AABBreduction(volatile int32_t* sMinX,
                                     volatile int32_t* sMaxX,
                                     volatile int32_t* sMinY,
                                     volatile int32_t* sMaxY,
                                     int32_t* d_AABB,
                                     const uint32_t& pIdx)
{

    //if blockDim>=512
    if(threadIdx.x<256)
    {
        sMinX[threadIdx.x] = min(sMinX[threadIdx.x],sMinX[threadIdx.x+256]);
        sMaxX[threadIdx.x] = max(sMaxX[threadIdx.x],sMaxX[threadIdx.x+256]);
        sMinY[threadIdx.x] = min(sMinY[threadIdx.x],sMinY[threadIdx.x+256]);
        sMaxY[threadIdx.x] = max(sMaxY[threadIdx.x],sMaxY[threadIdx.x+256]);
    }__syncthreads();
    //if blockDim>=256
    if(threadIdx.x<128)
    {
        sMinX[threadIdx.x] = min(sMinX[threadIdx.x],sMinX[threadIdx.x+128]);
        sMaxX[threadIdx.x] = max(sMaxX[threadIdx.x],sMaxX[threadIdx.x+128]);
        sMinY[threadIdx.x] = min(sMinY[threadIdx.x],sMinY[threadIdx.x+128]);
        sMaxY[threadIdx.x] = max(sMaxY[threadIdx.x],sMaxY[threadIdx.x+128]);
    }__syncthreads();
    //if blockDim>=128
    if(threadIdx.x<64)
    {
        sMinX[threadIdx.x] = min(sMinX[threadIdx.x],sMinX[threadIdx.x+64]);
        sMaxX[threadIdx.x] = max(sMaxX[threadIdx.x],sMaxX[threadIdx.x+64]);
        sMinY[threadIdx.x] = min(sMinY[threadIdx.x],sMinY[threadIdx.x+64]);
        sMaxY[threadIdx.x] = max(sMaxY[threadIdx.x],sMaxY[threadIdx.x+64]);
    }__syncthreads();

    //WARP reduction no need for synchronization !!
    if(threadIdx.x<32)
    {
        //if blockDim>=64
        sMinX[threadIdx.x] = min(sMinX[threadIdx.x],sMinX[threadIdx.x+32]);
        sMaxX[threadIdx.x] = max(sMaxX[threadIdx.x],sMaxX[threadIdx.x+32]);
        sMinY[threadIdx.x] = min(sMinY[threadIdx.x],sMinY[threadIdx.x+32]);
        sMaxY[threadIdx.x] = max(sMaxY[threadIdx.x],sMaxY[threadIdx.x+32]);
        //if blockDim>=32
        sMinX[threadIdx.x] = min(sMinX[threadIdx.x],sMinX[threadIdx.x+16]);
        sMaxX[threadIdx.x] = max(sMaxX[threadIdx.x],sMaxX[threadIdx.x+16]);
        sMinY[threadIdx.x] = min(sMinY[threadIdx.x],sMinY[threadIdx.x+16]);
        sMaxY[threadIdx.x] = max(sMaxY[threadIdx.x],sMaxY[threadIdx.x+16]);
        //if blockDim>=16
        sMinX[threadIdx.x] = min(sMinX[threadIdx.x],sMinX[threadIdx.x+8]);
        sMaxX[threadIdx.x] = max(sMaxX[threadIdx.x],sMaxX[threadIdx.x+8]);
        sMinY[threadIdx.x] = min(sMinY[threadIdx.x],sMinY[threadIdx.x+8]);
        sMaxY[threadIdx.x] = max(sMaxY[threadIdx.x],sMaxY[threadIdx.x+8]);
        //if blockDim>=8
        sMinX[threadIdx.x] = min(sMinX[threadIdx.x],sMinX[threadIdx.x+4]);
        sMaxX[threadIdx.x] = max(sMaxX[threadIdx.x],sMaxX[threadIdx.x+4]);
        sMinY[threadIdx.x] = min(sMinY[threadIdx.x],sMinY[threadIdx.x+4]);
        sMaxY[threadIdx.x] = max(sMaxY[threadIdx.x],sMaxY[threadIdx.x+4]);
        //if blockDim>=4
        sMinX[threadIdx.x] = min(sMinX[threadIdx.x],sMinX[threadIdx.x+2]);
        sMaxX[threadIdx.x] = max(sMaxX[threadIdx.x],sMaxX[threadIdx.x+2]);
        sMinY[threadIdx.x] = min(sMinY[threadIdx.x],sMinY[threadIdx.x+2]);
        sMaxY[threadIdx.x] = max(sMaxY[threadIdx.x],sMaxY[threadIdx.x+2]);
        //if blockDim>=2
        sMinX[threadIdx.x] = min(sMinX[threadIdx.x],sMinX[threadIdx.x+1]);
        sMaxX[threadIdx.x] = max(sMaxX[threadIdx.x],sMaxX[threadIdx.x+1]);
        sMinY[threadIdx.x] = min(sMinY[threadIdx.x],sMinY[threadIdx.x+1]);
        sMaxY[threadIdx.x] = max(sMaxY[threadIdx.x],sMaxY[threadIdx.x+1]);
    }

    //TODO: Always be sure threadIdx.x==0 is active !!
    if(threadIdx.x==0)//warp 0
    {
        /* use the first thread in the block to
         * save the partial result of that block
         * in order to get the full result out of the 6 blocks.
         * The partial result for each block is saved in sMXXX[0]
         * pIdx == idx particle [0,1023]
         */
//      d_AABB[blockIdx.x*Nparticle_+pIdx] = sMinX[0];
//      d_AABB[blockIdx.x*Nparticle_+pIdx + _6Nparticle_] = sMaxX[0];
//      d_AABB[blockIdx.x*Nparticle_+pIdx + _12Nparticle_] = sMinY[0];
//      d_AABB[blockIdx.x*Nparticle_+pIdx + _18Nparticle_] = sMaxY[0];


//      d_AABB[pIdx*_4fxptBlocks_  + blockIdx.x*4] = sMinX[0];
//      d_AABB[pIdx*_4fxptBlocks_  + blockIdx.x*4 + 1] = sMaxX[0];
//      d_AABB[pIdx*_4fxptBlocks_  + blockIdx.x*4 + 2] = sMinY[0];
//      d_AABB[pIdx*_4fxptBlocks_  + blockIdx.x*4 + 3] = sMaxY[0];

//      d_AABB[pIdx*_4fxptBlocks_  + blockIdx.x] = sMinX[0];
//      d_AABB[pIdx*_4fxptBlocks_  + blockIdx.x + 6] = sMaxX[0];
//      d_AABB[pIdx*_4fxptBlocks_  + blockIdx.x + 12] = sMinY[0];
//      d_AABB[pIdx*_4fxptBlocks_  + blockIdx.x + 18] = sMaxY[0];

        //TODO: OK with shuffle fcns reduction:
//      d_AABB[pIdx*_4warpSize_  + blockIdx.x] = sMinX[0];
//      d_AABB[pIdx*_4warpSize_  + blockIdx.x + 32] = sMaxX[0];
//      d_AABB[pIdx*_4warpSize_  + blockIdx.x + 64] = sMinY[0];
//      d_AABB[pIdx*_4warpSize_  + blockIdx.x + 96] = sMaxY[0];

        //OK with shared memory reduction
        d_AABB[pIdx*_8warpSize_  + blockIdx.x] = sMinX[0];
        d_AABB[pIdx*_8warpSize_  + blockIdx.x + 64] = sMaxX[0];
        d_AABB[pIdx*_8warpSize_  + blockIdx.x + 128] = sMinY[0];
        d_AABB[pIdx*_8warpSize_  + blockIdx.x + 192] = sMaxY[0];

//      if(pIdx==0)
//      {
//          printf("pIdx: %d, blockIdx.x: %d, sMinX[0]: %d, sMaxX[0]: %d, sMinY[0]: %d, sMaxY[0]: %d\n",
//                  pIdx,blockIdx.x,sMinX[0],sMaxX[0],sMinY[0],sMaxY[0]);
//      }

    }

    return;
}

__forceinline__ __device__ void renderFixedPointAABB(const float4& v1,
                                        const float4& v2,
                                        const float4& v3,
                                        int* d_depth_buffer,
                                        const uint32_t& offset_buffer,
                                        const uint32_t& tIdx,
                                        const uint32_t& pIdx,
                                        volatile int32_t* sMinX,
                                        volatile int32_t* sMaxX,
                                        volatile int32_t* sMinY,
                                        volatile int32_t* sMaxY,
                                        int32_t* d_AABB)
{

    // 28.4 fixed-point coordinates: 28 integer and 4 after the point
    //Scaling factor == 2^4 = 16
    //*ROUND* cast since +0.5f is already added, after the rototranslation, to UV vertices
//  const int Y1 = int(16.0f * v1.y);
//  const int Y2 = int(16.0f * v2.y);
//  const int Y3 = int(16.0f * v3.y);
//
//  const int X1 = int(16.0f * v1.x);
//  const int X2 = int(16.0f * v2.x);
//  const int X3 = int(16.0f * v3.x);


    //+0.5f is NOT added, after the rototranslation, to UV vertices
    const int Y1 = __float2int_rn(16.0f * v1.y);
    const int Y2 = __float2int_rn(16.0f * v2.y);
    const int Y3 = __float2int_rn(16.0f * v3.y);

    const int X1 = __float2int_rn(16.0f * v1.x);
    const int X2 = __float2int_rn(16.0f * v2.x);
    const int X3 = __float2int_rn(16.0f * v3.x);


    // Bounding rectangle
    //Scaling factor 1 = 2^0
    const int minx = (min(min(X1, X2), X3) + 0xF) >> 4;
    const int maxx = (max(max(X1, X2), X3) + 0xF) >> 4;
    const int miny = (min(min(Y1, Y2), Y3) + 0xF) >> 4;
    const int maxy = (max(max(Y1, Y2), Y3) + 0xF) >> 4;

    //I have 4 shared memories of 512 faces each per 6 blocks, hence 512*6~>numFaces
    //Init with all faces AABB
    sMinX[threadIdx.x] = minx;
    sMaxX[threadIdx.x] = maxx;
    sMinY[threadIdx.x] = miny;
    sMaxY[threadIdx.x] = maxy;
    __syncthreads();

    AABBreduction(sMinX,sMaxX,sMinY,sMaxY,d_AABB,pIdx);

    // Deltas
    //Scaling factor == 2^4 = 16
    //B
    const int DX12 = X1 - X2;
    const int DX23 = X2 - X3;
    const int DX31 = X3 - X1;
    //A
    const int DY12 = Y1 - Y2;
    const int DY23 = Y2 - Y3;
    const int DY31 = Y3 - Y1;

    //Area with scaling factor 16*16=2^4*2^4 == 2^8
    int triArea = /*std::abs*/(DX12 * DY31 - DX31 * DY12);
    if(triArea<=0)
    {
        return;
    }
    //AABB zero Area
    if((maxx-minx)<=0 || (maxy-miny)<=0)
        return;

    const float _1triArea = overA_*__frcp_rn(__int2float_rn(triArea));//__fdividef(overA_,__int2float_rn(triArea));

//  printf("triArea: %d\n",triArea);
//  printf("__int2float_rn(triArea): %f\n",__int2float_rn(triArea));
//  printf("__frcp_rn(__int_as_float(triArea)): %f\n",__frcp_rn(__int_as_float(triArea)));
//  printf("_1triArea: %f\n",_1triArea);
    // Fixed-point deltas
    // Scaling factor == 2^8 = 2^4*2^4 = 256
    //B
    const int FDX12 = DX12 << 4;
    const int FDX23 = DX23 << 4;
    const int FDX31 = DX31 << 4;
    //A
    const int FDY12 = DY12 << 4;
    const int FDY23 = DY23 << 4;
    const int FDY31 = DY31 << 4;

    // Half-edge constants
    //Scaling Factor 16*16=2^4*2^4 == 2^8
    int C1 = DY12 * X1 - DX12 * Y1;
    int C2 = DY23 * X2 - DX23 * Y2;
    int C3 = DY31 * X3 - DX31 * Y3;
    // Correct for fill convention
    //Only top-left edges are considered inside the triangle
    //so we can leave the test if(CX1 > 0 && CX2 > 0 && CX3 > 0) strictly greater than 0 without drawing non top-left edges.
    //here +1 is in Scaling factor 2^8 so real is +1 >> 8
    //add +1 without if
    C1 += (DY12 < 0 || (DY12 == 0 && DX12 > 0));
    C2 += (DY23 < 0 || (DY23 == 0 && DX23 > 0));
    C3 += (DY31 < 0 || (DY31 == 0 && DX31 > 0));

    //Scaling factor 2^4
    const int MINY = (miny << 4);
    const int MINX = (minx << 4);
    //Scaling factor 2^8
    int CY1 = C1 + DX12 * MINY - DY12 * MINX;
    int CY2 = C2 + DX23 * MINY - DY23 * MINX;
    int CY3 = C3 + DX31 * MINY - DY31 * MINX;

    //Scaling Factor 2^8
    //float z1 = v1.z;
    const float z2 = (v2.z - v1.z)* _1triArea ;
    const float z3 = (v3.z - v1.z)* _1triArea ;

    for(int y = miny; y < maxy; ++y)
    {
        //Scaling factor 2^8
        int CX1 = CY1;
        int CX2 = CY2;
        int CX3 = CY3;

        for(int x = minx; x < maxx; ++x)
        {

            //Test Pixel inside triangle
            //const int32_t cx1 = (CX1+ 0xF)>>8;
            const int32_t cx2 = (CX2 + 0xF)>>8;
            const int32_t cx3 = (CX3 + 0xF)>>8;

            //mask == 1 if >0; 0 otherwise
            int32_t mask =( ( !((CX1 & 0x80000000) | !CX1)) & ( !((CX2 & 0x80000000) | !CX2)) & ( !((CX3 & 0x80000000) | !CX3))) > 0;

            const int idx = (y*cols_ + x) + offset_buffer;

            float depth = v1.z;
            depth += cx2*z2;
            depth += cx3*z3;

            int depth_int = __float2int_rn(depth);
            //float depth = v1.z + cx2*z2 + cx3*z3;
//          float depth = __fmaf_rn(__int2float_rn(cx2),z2,v1.z);
//          depth = __fmaf_rn(__int2float_rn(cx3),z3,depth);

            //float previousDepthValue = d_depth_buffer[idx];
//          int previousDepthValue = d_depth_buffer[idx];

            /* Up to now, it costs lesser a DepthBuffer writing branch than a DepthBuffer writing mask.
             * The Time of a single kernel call drop down to 0.055ms wrt to 0.07ms as before, but
             * the time of 1024 streamed call is left almost unchanged ~17.09-17.19 ms.
             * Using 6 blocks per 512 ths we get improvement of ~0.6ms wrt 12 blks per 256 ths
             */
//          float maxdepth = min(depth, previousDepthValue);
//          depth = (mask & 0x80000000) ? previousDepthValue : maxdepth;
//          d_depth_buffer[idx] = depth;

            /*TODO: Solve memory coalesced problem and
             * the Race condition one to speed things up !
             */
//          if(!(mask & 0x80000000) && depth_int<previousDepthValue)
//          {
//              //printf("depth: %f\n",depth);
//              d_depth_buffer[idx] = depth_int;
//          }
            if(!(mask & 0x80000000))
            {
                    atomicMin(&(d_depth_buffer[idx]), depth_int);
            }

            //Scaling factor 2^8
            CX1 -= FDY12;
            CX2 -= FDY23;
            CX3 -= FDY31;
        }
        //Scaling factor 2^8
        CY1 += FDX12;
        CY2 += FDX23;
        CY3 += FDX31;

    }

    return;
}

template <class T>
__forceinline__ __device__ void FitnessReductionMultiBlock(volatile T* sError, const int32_t& tIdx)
{

    //sError is 2048 elements length
    //tIdx [0-1023] is 1024 threads
    //if(tIdx<1024) { sError[tIdx] += sError[tIdx+1024]; }__syncthreads();
    if(threadIdx.x<512)  { sError[threadIdx.x] += sError[threadIdx.x+512]; }__syncthreads();
    if(threadIdx.x<256)  { sError[threadIdx.x] += sError[threadIdx.x+256]; }__syncthreads();
    if(threadIdx.x<128)  { sError[threadIdx.x] += sError[threadIdx.x+128]; }__syncthreads();
    if(threadIdx.x<64)   { sError[threadIdx.x] += sError[threadIdx.x+64]; }__syncthreads();

    if(threadIdx.x<32)//WARP REDUCE no need for synchronization
    {
        sError[threadIdx.x] += sError[threadIdx.x+32];
        sError[threadIdx.x] += sError[threadIdx.x+16];
        sError[threadIdx.x] += sError[threadIdx.x+8];
        sError[threadIdx.x] += sError[threadIdx.x+4];
        sError[threadIdx.x] += sError[threadIdx.x+2];
        sError[threadIdx.x] += sError[threadIdx.x+1];
    }

    //sError[0] holds the total Fitness error for that particle pIdx


}

//
template <class T>
__forceinline__ __device__ void FitnessReductionBallot(volatile T* sError,
        volatile unsigned int* sNrenderedPointsWarpBallot,
        volatile unsigned int* sNumSegDepthWoutValid3dModelPixel,
        volatile unsigned int* sNumRenderedDepthWoutValid3dClusterPixel, const int32_t& tIdx)
{

    //sError is 2048 elements length
    //tIdx [0-1023] is 1024 threads
    if(tIdx<1024) { sError[tIdx] += sError[tIdx+1024]; sNumSegDepthWoutValid3dModelPixel[tIdx] += sNumSegDepthWoutValid3dModelPixel[tIdx+1024]; sNumRenderedDepthWoutValid3dClusterPixel[tIdx] += sNumRenderedDepthWoutValid3dClusterPixel[tIdx+1024];}__syncthreads();
    if(tIdx<512)  { sError[tIdx] += sError[tIdx+512];  sNumSegDepthWoutValid3dModelPixel[tIdx] += sNumSegDepthWoutValid3dModelPixel[tIdx+512];  sNumRenderedDepthWoutValid3dClusterPixel[tIdx] += sNumRenderedDepthWoutValid3dClusterPixel[tIdx+512];}__syncthreads();
    if(tIdx<256)  { sError[tIdx] += sError[tIdx+256];  sNumSegDepthWoutValid3dModelPixel[tIdx] += sNumSegDepthWoutValid3dModelPixel[tIdx+256];  sNumRenderedDepthWoutValid3dClusterPixel[tIdx] += sNumRenderedDepthWoutValid3dClusterPixel[tIdx+256];}__syncthreads();
    if(tIdx<128)  { sError[tIdx] += sError[tIdx+128];  sNumSegDepthWoutValid3dModelPixel[tIdx] += sNumSegDepthWoutValid3dModelPixel[tIdx+128];  sNumRenderedDepthWoutValid3dClusterPixel[tIdx] += sNumRenderedDepthWoutValid3dClusterPixel[tIdx+128];}__syncthreads();
    if(tIdx<64)   { sError[tIdx] += sError[tIdx+64];   sNumSegDepthWoutValid3dModelPixel[tIdx] += sNumSegDepthWoutValid3dModelPixel[tIdx+64];   sNumRenderedDepthWoutValid3dClusterPixel[tIdx] += sNumRenderedDepthWoutValid3dClusterPixel[tIdx+64];}__syncthreads();

    if(tIdx<32)//WARP REDUCE no need for synchronization
    {
        sError[tIdx] += sError[tIdx+32];
        sNrenderedPointsWarpBallot[tIdx] += sNrenderedPointsWarpBallot[tIdx+32];
        sNumSegDepthWoutValid3dModelPixel[tIdx] += sNumSegDepthWoutValid3dModelPixel[tIdx+32];
        sNumRenderedDepthWoutValid3dClusterPixel[tIdx] += sNumRenderedDepthWoutValid3dClusterPixel[tIdx+32];

        sError[tIdx] += sError[tIdx+16];
        sNrenderedPointsWarpBallot[tIdx] += sNrenderedPointsWarpBallot[tIdx+16];
        sNumSegDepthWoutValid3dModelPixel[tIdx] += sNumSegDepthWoutValid3dModelPixel[tIdx+16];
        sNumRenderedDepthWoutValid3dClusterPixel[tIdx] += sNumRenderedDepthWoutValid3dClusterPixel[tIdx+16];

        sError[tIdx] += sError[tIdx+8];
        sNrenderedPointsWarpBallot[tIdx] += sNrenderedPointsWarpBallot[tIdx+8];
        sNumSegDepthWoutValid3dModelPixel[tIdx] += sNumSegDepthWoutValid3dModelPixel[tIdx+8];
        sNumRenderedDepthWoutValid3dClusterPixel[tIdx] += sNumRenderedDepthWoutValid3dClusterPixel[tIdx+8];

        sError[tIdx] += sError[tIdx+4];
        sNrenderedPointsWarpBallot[tIdx] += sNrenderedPointsWarpBallot[tIdx+4];
        sNumSegDepthWoutValid3dModelPixel[tIdx] += sNumSegDepthWoutValid3dModelPixel[tIdx+4];
        sNumRenderedDepthWoutValid3dClusterPixel[tIdx] += sNumRenderedDepthWoutValid3dClusterPixel[tIdx+4];

        sError[tIdx] += sError[tIdx+2];
        sNrenderedPointsWarpBallot[tIdx] += sNrenderedPointsWarpBallot[tIdx+2];
        sNumSegDepthWoutValid3dModelPixel[tIdx] += sNumSegDepthWoutValid3dModelPixel[tIdx+2];
        sNumRenderedDepthWoutValid3dClusterPixel[tIdx] += sNumRenderedDepthWoutValid3dClusterPixel[tIdx+2];

        sError[tIdx] += sError[tIdx+1];
        sNrenderedPointsWarpBallot[tIdx] += sNrenderedPointsWarpBallot[tIdx+1];
        sNumSegDepthWoutValid3dModelPixel[tIdx] += sNumSegDepthWoutValid3dModelPixel[tIdx+1];
        sNumRenderedDepthWoutValid3dClusterPixel[tIdx] += sNumRenderedDepthWoutValid3dClusterPixel[tIdx+1];
    }

    //sError[0] holds the total Fitness error for that particle pIdx
    //sNrenderedPointsWarpBallot[0] holds the Number of rendered pixels of the particle pIdx


}
//

//
template <class T>
__forceinline__ __device__ void FitnessReduction(volatile T* sError, volatile unsigned int* sNrenderedPoints, const int32_t& tIdx)
{

    //sError is 2048 elements length
    //tIdx [0-1023] is 1024 threads
    if(tIdx<1024) { sError[tIdx] += sError[tIdx+1024]; sNrenderedPoints[tIdx] += sNrenderedPoints[tIdx+1024];}__syncthreads();
    if(tIdx<512)  { sError[tIdx] += sError[tIdx+512];  sNrenderedPoints[tIdx] += sNrenderedPoints[tIdx+512];}__syncthreads();
    if(tIdx<256)  { sError[tIdx] += sError[tIdx+256];  sNrenderedPoints[tIdx] += sNrenderedPoints[tIdx+256];}__syncthreads();
    if(tIdx<128)  { sError[tIdx] += sError[tIdx+128];  sNrenderedPoints[tIdx] += sNrenderedPoints[tIdx+128];}__syncthreads();
    if(tIdx<64)   { sError[tIdx] += sError[tIdx+64];   sNrenderedPoints[tIdx] += sNrenderedPoints[tIdx+64];}__syncthreads();

    if(tIdx<32)//WARP REDUCE no need for synchronization
    {
        sError[tIdx] += sError[tIdx+32];
        sNrenderedPoints[tIdx] += sNrenderedPoints[tIdx+32];

        sError[tIdx] += sError[tIdx+16];
        sNrenderedPoints[tIdx] += sNrenderedPoints[tIdx+16];

        sError[tIdx] += sError[tIdx+8];
        sNrenderedPoints[tIdx] += sNrenderedPoints[tIdx+8];

        sError[tIdx] += sError[tIdx+4];
        sNrenderedPoints[tIdx] += sNrenderedPoints[tIdx+4];

        sError[tIdx] += sError[tIdx+2];
        sNrenderedPoints[tIdx] += sNrenderedPoints[tIdx+2];

        sError[tIdx] += sError[tIdx+1];
        sNrenderedPoints[tIdx] += sNrenderedPoints[tIdx+1];
    }

    //sError[0] holds the total Fitness error for that particle pIdx
    //sNrenderedPoints[0] holds the Number of rendered pixels of the particle pIdx


}
//

__forceinline__ __device__ unsigned int fixedpoint_modulo(unsigned int &x, unsigned int &n )

{

    unsigned int k, r ;

    //unsigned int n = 640;//256 ;

    unsigned int one_over_n = __frcp_rn(__uint2float_rn(n))*_2_p32_;//0x00666666;//0x01000000; // 1/n * 2^32


    k = __umulhi(x, one_over_n);

    r = x - n*k ;

    return r ;

}

///
__forceinline__ __device__ unsigned int double_modulo(const unsigned int &x, const unsigned int &n )
{
    double tmp1 ;

    unsigned int k, r ;

    double one_over_n = __frcp_rn(__uint2double_rn(n));

    tmp1 = __uint2double_rn( x ) ;

    tmp1 = tmp1 * one_over_n ;

    k = __double2uint_rz( tmp1 ) ;

    r = x - n*k ;

    return r ;

}


///
__forceinline__ __host__ void saveDepthRenderedCSVBestParticle(const std::string& path, const int* depth, const int& tIdx)
{
    std::ofstream ofs(path.c_str(), std::ofstream::out);

    unsigned int real_pxInx=0;
    for(int pxIdx=0;pxIdx<RxC_;++pxIdx)
    {
        int index = tIdx*RxC_ + pxIdx;
        ofs<<depth[index];
        if(real_pxInx==cols_-1)
        {
            ofs<<"\n";
            real_pxInx=0;
        }
        else
        {
            ofs<<", ";
            ++real_pxInx;
        }
    }
    ofs.close();
}
///

__forceinline__ __host__ void saveDepthRenderedCSV(const std::string& path, const int* depth)
{
    std::ofstream ofs(path.c_str(), std::ofstream::out);

    for(int tIdx=0;tIdx<1;++tIdx)//only rendered of first particle
    {
        unsigned int real_pxInx=0;
        for(int pxIdx=0;pxIdx<RxC_;++pxIdx)
        {
            int index = tIdx*RxC_ + pxIdx;
            ofs<<depth[index];
            if(real_pxInx==cols_-1)
            {
                ofs<<"\n";
                real_pxInx=0;
            }
            else
            {
                ofs<<", ";
                ++real_pxInx;
            }
        }

    }
    ofs.close();
}

__forceinline__ __host__ void saveDepthRenderedCSVAllParticle(const std::string& path, const float* depth)
{
    std::ofstream ofs (path.c_str(), std::ofstream::out);


    for(int tIdx=0;tIdx<1;++tIdx)//only rendered of first particle
    {
        unsigned int real_pxInx=0;
        for(int pxIdx=0;pxIdx<RxC_;++pxIdx)
        {
            int index = tIdx + pxIdx*Nparticle_;
            ofs<<depth[index];
            if(real_pxInx==cols_-1)
            {
                ofs<<"\n";
                real_pxInx=0;
            }
            else
            {
                ofs<<", ";
                ++real_pxInx;
            }
        }

    }
    ofs.close();
}

__forceinline__ __host__ __device__ void printVector(const float* w, int size)
{
    printf("[ ");
    for (int i=0; i<size; ++i) {
        if (i==size-1) {
            printf("%f ",w[i]);
        }
        else
            printf("%f, ",w[i]);
    }
    printf("]\n");
}
__forceinline__ __host__ void printVector(const int* w, int size)
{
    printf("[ ");
    for (int i=0; i<size; ++i) {
        if (i==size-1) {
            printf("%d ",w[i]);
        }
        else
            printf("%d, ",w[i]);
    }
    printf("]\n");
}


__forceinline__ __host__ void printVector(const unsigned int* w, int size)
{
    printf("[ ");
    for (int i=0; i<size; ++i) {
        if (i==size-1) {
            printf("%u ",w[i]);
        }
        else
            printf("%u, ",w[i]);
    }
    printf("]\n");
}

template <typename T>
__forceinline__ std::string NumberToString ( T Number )
{
 std::ostringstream ss;
 ss << Number;
 return ss.str();
}

__forceinline__ __host__ void printOneParticlePoseFromGlobalMem(const float* h_pso_pos, const int& tIdx, float* pose)
{
    const uint32_t idx_tx = tIdx + tx_*Nparticle_;
    const uint32_t idx_ty = tIdx + ty_*Nparticle_;
    const uint32_t idx_tz = tIdx + tz_*Nparticle_;
    const uint32_t idx_q0 = tIdx + q0_*Nparticle_;
    const uint32_t idx_q1 = tIdx + q1_*Nparticle_;
    const uint32_t idx_q2 = tIdx + q2_*Nparticle_;
    const uint32_t idx_q3 = tIdx + q3_*Nparticle_;

    pose[tx_] = h_pso_pos[idx_tx];
    pose[ty_] = h_pso_pos[idx_ty];
    pose[tz_] = h_pso_pos[idx_tz];
    pose[q0_] = h_pso_pos[idx_q0];
    pose[q1_] = h_pso_pos[idx_q1];
    pose[q2_] = h_pso_pos[idx_q2];
    pose[q3_] = h_pso_pos[idx_q3];

    printf("Particle Idx: %d pose\n",tIdx);
    printVector(pose,Ndim_);


    return;
}

__forceinline__ __host__ int iround(const float& val)
{
    return ( (int)( val + 0.5f) );
}

template <class T>
inline T min3(const T &a, const T &b, const T &c)
{ return std::min(a, std::min(b, c)); }

template <class T>
inline T max3(const T &a, const T &b, const T &c)
{ return std::max(a, std::max(b, c)); }

__host__ __forceinline__
void projectModelPointsToPixelsCPU(const float* pso_pose_vec, const float4& point, float4& UVz)
{
    /* input:
     * @pso_pose_vec: is the particle state vector:
     *                pose[tx,ty,tz,q0,q1,q2,q3] in [meter, unit_quat]
     * @point: is a 3D point [in meter] of type float4 with .w not considered
     * output:
     * @UVz: is the normalized pixel coordinate UV of
     *       the projected point and its depth value Z in millimeters (but all in float).
     */
    /* UVZ = K*T*XYZ; UV = UVZ/Z; */
    //Rotation
    float pprime[3];
    quatVectRotationPSOstateVect(pso_pose_vec,point,&pprime[0]);
    //Translation
    pprime[tx_] += pso_pose_vec[tx_];
    pprime[ty_] += pso_pose_vec[ty_];
    pprime[tz_] += pso_pose_vec[tz_];
    //Projection K and normalize
    /*  |cx*z + fx*x    |       |cx + fx*(x/z)|
        |cy*z + fy*y|  ==>  |cy + fy*(y/z)|
        |z          |       |     1       |
     */

    //USING MACROs we gain 20 ms over __constant__
    const float _1pprimeZ = 1.f/pprime[tz_];
    UVz.x = ( CX + FX*(pprime[tx_]*_1pprimeZ)); //+ tocast_ );
    UVz.y = ( CY + FY*(pprime[ty_]*_1pprimeZ)); //+ tocast_ );
    UVz.z = (tomm_*pprime[tz_]); //+ tocast_);//in mm
    UVz.w = 0.f;
    return;
}

__forceinline__ __host__ void CPUfixedPointRender(float* pDepthBuffer,
                                                 const float4& v1,
                                                 const float4& v2,
                                                 const float4& v3)
{

    // 28.4 fixed-point coordinates: 28 integer and 4 after the point
    //Scaling factor == 2^4 = 16
    const int Y1 = iround(16.0f * v1.y);
    const int Y2 = iround(16.0f * v2.y);
    const int Y3 = iround(16.0f * v3.y);

    const int X1 = iround(16.0f * v1.x);
    const int X2 = iround(16.0f * v2.x);
    const int X3 = iround(16.0f * v3.x);

    // Deltas
    //Scaling factor == 2^4 = 16
    //B
    const int DX12 = X1 - X2;
    const int DX23 = X2 - X3;
    const int DX31 = X3 - X1;
    //A
    const int DY12 = Y1 - Y2;
    const int DY23 = Y2 - Y3;
    const int DY31 = Y3 - Y1;

    //Area with scaling factor 16*16=2^4*2^4 == 2^8
    //printf("%d\n",DX12 * DY31 - DX31 * DY12);
    int triArea = /*std::abs*/(DX12 * DY31 - DX31 * DY12);
    //triArea = (triArea + 0xF) >> 8;
    if(triArea<=0)
    {
        return;
    }
    float _1triArea = overA_/float(triArea);
    // Fixed-point deltas
    // Scaling factor == 2^8 = 2^4*2^4 = 256
    //B
    const int FDX12 = DX12 << 4;
    const int FDX23 = DX23 << 4;
    const int FDX31 = DX31 << 4;
    //A
    const int FDY12 = DY12 << 4;
    const int FDY23 = DY23 << 4;
    const int FDY31 = DY31 << 4;

    // Bounding rectangle
    //Scaling factor 1 = 2^0
    int minx = (min3(X1, X2, X3) + 0xF) >> 4;
    int maxx = (max3(X1, X2, X3) + 0xF) >> 4;
    int miny = (min3(Y1, Y2, Y3) + 0xF) >> 4;
    int maxy = (max3(Y1, Y2, Y3) + 0xF) >> 4;

    //AABB zero Area
    if((maxx-minx)<=0 || (maxy-miny)<=0)
        return;

    // Half-edge constants
    //Sacaling Factor 16*16=2^4*2^4 == 2^8
    int C1 = DY12 * X1 - DX12 * Y1;
    int C2 = DY23 * X2 - DX23 * Y2;
    int C3 = DY31 * X3 - DX31 * Y3;

    // Correct for fill convention
    //Only top-left edges are considered inside the triangle
    //so we can leave the test if(CX1 > 0 && CX2 > 0 && CX3 > 0) strictly greater than 0 without drawing non top-left edges.
    //here +1 is in Scaling factor 2^8 so real is +1 >> 8
//    if(DY12 < 0 || (DY12 == 0 && DX12 > 0)) ++C1;
//    if(DY23 < 0 || (DY23 == 0 && DX23 > 0)) ++C2;
//    if(DY31 < 0 || (DY31 == 0 && DX31 > 0)) ++C3;
    //add +1 without if
    C1 += (DY12 < 0 || (DY12 == 0 && DX12 > 0));
    C2 += (DY23 < 0 || (DY23 == 0 && DX23 > 0));
    C3 += (DY31 < 0 || (DY31 == 0 && DX31 > 0));

    //Scaling factor 2^4
    const int MINY = (miny << 4);
    const int MINX = (minx << 4);
    //Scaling factor 2^8
    int CY1 = C1 + DX12 * MINY - DY12 * MINX;
    int CY2 = C2 + DX23 * MINY - DY23 * MINX;
    int CY3 = C3 + DX31 * MINY - DY31 * MINX;

    float z2 = (v2.z - v1.z)* _1triArea ;
    float z3 = (v3.z - v1.z)* _1triArea ;

    for(int y = miny; y < maxy; ++y)
    {
        //Scaling factor 2^8
        int CX1 = CY1;
        int CX2 = CY2;
        int CX3 = CY3;

        for(int x = minx; x < maxx; ++x)
        {

            //Test Pixel inside triangle
            //const int32_t cx1 = (CX1+ 0xF)>>8;
            const int32_t cx2 = (CX2 + 0xF)>>8;
            const int32_t cx3 = (CX3 + 0xF)>>8;
            //int mask = fxptZero < ( (CX1+ 0xF)>>8 | cx2 | cx3 ); //? 1 : 0;
            //mask == 1 if >0; 0 otherwise
            int32_t mask =( ( !((CX1 & 0x80000000) | !CX1)) & ( !((CX2 & 0x80000000) | !CX2)) & ( !((CX3 & 0x80000000) | !CX3))) > 0;

            int idx = y*cols_ + x;

            float depth = v1.z;
            depth += cx2*z2;
            depth += cx3*z3;


            float previousDepthValue = pDepthBuffer[idx];

            if(!(mask & 0x80000000) && depth<previousDepthValue)
            {
                pDepthBuffer[idx] = depth;
            }

            //Scaling factor 2^8
            CX1 -= FDY12;
            CX2 -= FDY23;
            CX3 -= FDY31;
        }
        //Scaling factor 2^8
        CY1 += FDX12;
        CY2 += FDX23;
        CY3 += FDX31;

    }



    return;
}

template<class T>
__forceinline__ void saveBestParticleRender2MatlabCSV(const T* mat, std::string path="/Users/giorgio/Documents/MATLAB/mypc.csv")
{
    //save Depth mm for float* in csv format
    std::ofstream ofs (path, std::ofstream::out);
    for (int r=0; r<rows_; ++r) {
        for (int c=0; c<cols_; ++c)
        {
            const int idx = r*cols_ + c;
            ofs<<mat[idx];
            if (c < cols_-1) {
                ofs<<", ";
            }
        }
        ofs<<"\n";
    }
    ofs.close();
}

__forceinline__ __host__ void RenderCPU(float* pDepthBuffer, const float* pso_pose_vec ,
                                const std::vector<float>& ord_obj_vect, const int& numFaces,
                                const std::string& path_to_save_csv)
{


    for(int i=0;i<numFaces;++i)
    {

        float4 v[3];
        float4 UVz[3];
        for(int j=0;j<3;++j)//three vertices of the face
        {


            const float x = ord_obj_vect[i + 0 + j*numFaces];
            const float y = ord_obj_vect[i + 3*numFaces + j*numFaces];
            const float z = ord_obj_vect[i + 6*numFaces + j*numFaces];
            v[j] = make_float4(x,y,z,0);

            projectModelPointsToPixelsCPU(pso_pose_vec,v[j],UVz[j]);

        }

        CPUfixedPointRender(pDepthBuffer,UVz[0],UVz[1],UVz[2]);
    }


    //Save the Depth map in .csv
    saveBestParticleRender2MatlabCSV(pDepthBuffer,path_to_save_csv);

}

#endif /* MAIN_H_ */
