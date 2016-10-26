/*
 * main.cpp
 *
 *  Created on: 02/gen/2016
 *      Author: giorgio
 */


#include <cuda_runtime.h>
#include "handle_error.h"
//#include <random> //c++11
#include <gsl/gsl_rng.h>
#include <vector>
 #include "vector_types.h"
#include "main.h"
//#include "shampoosegdepth.h"
//#include "shampooC.h"
#include "shampoo1024faces.h"
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/distance.h>

#include "GraphCannySeg.h"



//#include <sys/time.h> // for gettimeofday()
#include "myTimer.h"


// for loading ACCV dataset
cv::Mat loadDepth( std::string a_name )
{
    cv::Mat lp_mat;
    std::ifstream l_file(a_name.c_str(),std::ofstream::in|std::ofstream::binary );
    
    if( l_file.fail() == true )
    {
        printf("cv_load_depth: could not open file for writing!\n");
        return lp_mat;
    }
    int l_row;
    int l_col;
    
    l_file.read((char*)&l_row,sizeof(l_row));
    l_file.read((char*)&l_col,sizeof(l_col));
    
    IplImage * lp_image = cvCreateImage(cvSize(l_col,l_row),IPL_DEPTH_16U,1);
    
    for(int l_r=0;l_r<l_row;++l_r)
    {
        for(int l_c=0;l_c<l_col;++l_c)
        {
            l_file.read((char*)&CV_IMAGE_ELEM(lp_image,unsigned short,l_r,l_c),sizeof(unsigned short));
        }
    }
    l_file.close();
    
    lp_mat= cv::Mat(lp_image);
    return lp_mat;
}



//texture<float4, 1, cudaReadModeElementType> g_ord_obj_float4Texture;
//texture<int, 1, cudaReadModeElementType> textureDepthKinect;

/* KERNEL INIT PSO */
__global__ void InitQPSO(   float* d_pso_pos_nb,//global best pose 1xDim if global topology
                            float* d_randGen,
                            float* d_pso_pos, //particles current pose [Ndim*Nparticle]
                            float* d_pso_vel, //particles current vels [Ndim*Nparticle]
                            float* d_pso_pos_b, //particles best pose  [Ndim*Nparticle]
                            unsigned int* d_randIdx) //to keep track of the index each thread has within the d_randGen vector
{
    // For Each particle tIdx !!! We assume we have one thread per pso particle
    int tIdx = blockDim.x * blockIdx.x + threadIdx.x;

//  int x = threadIdx.x + blockIdx.x * blockDim.x;
//  int y = threadIdx.y + blockIdx.y * blockDim.y;
//  int tIdx = x + y * blockDim.x * gridDim.x;

    unsigned int s_randIdx=0;//this should be less than strideDimRandGem_ to avoid some thread
                        //gets the same random value of the others. This means one thread
                        //should read less then strideDimRandGem_ random numbers

    /* Try to access the Random Generator Vector in a coalesced fashion... */
    //d_randGen[tIdx+(s_randIdx++)*Nparticle_];

    /* PARTICLE tIdx INITIALIZATION */
    /*
     * pso_pos[Ndim] actual particle pose
     * pso_vel[Ndim] actual particle lin. and ang. velocity
     * pso_pos_b[Ndim] actual particle personal best pose
     */
    float pso_pos[Ndim_];
    float pso_vel[Ndim_];
    float pso_pos_b[Ndim_];

    // TRANSLATION tx pos & vel
    pso_pos[tx_] = d_qpso_params[0].x_lo[tx_] +
            (d_qpso_params[0].x_hi[tx_]-d_qpso_params[0].x_lo[tx_])*d_randGen[tIdx+(s_randIdx++)*Nparticle_];
    pso_pos_b[tx_] = pso_pos[tx_];
    pso_vel[tx_] = -vdot_ + (2.f*vdot_)*d_randGen[tIdx+(s_randIdx++)*Nparticle_];

    // TRANSLATION ty pos & vel
    pso_pos[ty_] = d_qpso_params[0].x_lo[ty_] +
            (d_qpso_params[0].x_hi[ty_]-d_qpso_params[0].x_lo[ty_])*d_randGen[tIdx+(s_randIdx++)*Nparticle_];
    pso_pos_b[ty_] = pso_pos[ty_];
    pso_vel[ty_] = -vdot_ + (2.f*vdot_)*d_randGen[tIdx+(s_randIdx++)*Nparticle_];

    // TRANSLATION tz pos & vel
    pso_pos[tz_] = d_qpso_params[0].x_lo[tz_] +
            (d_qpso_params[0].x_hi[tz_]-d_qpso_params[0].x_lo[tz_])*d_randGen[tIdx+(s_randIdx++)*Nparticle_];
    pso_pos_b[tz_] = pso_pos[tz_];
    pso_vel[tz_] = -vdot_ + (2.f*vdot_)*d_randGen[tIdx+(s_randIdx++)*Nparticle_];


    // QUATERNION INIT
    //Perturbate the INIT Quaternion
    float qinit[Ndim_] = //{0.f,0.f,0.f,1.f,0.f,0.f,0.f};
                        //{0.f,0.f,0.f, 0.f, 0.f, 0.7071f,-0.7071f};
                        //{0.,0.,0.,0.7071,0.,0.7071,0.};
                        {0.f,0.f,0.f, 0.7071f,0.7071f,0.f,0.f};
                        //{0.,0.,0.,0.7071,0.,0.,0.7071};

    float wpert[Ndim_] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    wpert[q1_] = aminX_ + (amaxX_-aminX_)*d_randGen[tIdx+(s_randIdx++)*Nparticle_];
    wpert[q2_] = aminY_ + (amaxY_-aminY_)*d_randGen[tIdx+(s_randIdx++)*Nparticle_];
    wpert[q3_] = aminZ_ + (amaxZ_-aminZ_)*d_randGen[tIdx+(s_randIdx++)*Nparticle_];

    QuatKinematicsPSOStateVectWithInit(qinit,pso_pos,wpert);
    //copy result to personal best
    pso_pos_b[q0_] = pso_pos[q0_];
    pso_pos_b[q1_] = pso_pos[q1_];
    pso_pos_b[q2_] = pso_pos[q2_];
    pso_pos_b[q3_] = pso_pos[q3_];
    //Generate an Inital Random Angular Velocity for each particle
    pso_vel[q0_] = 0.f;
    pso_vel[q1_] = aminX_ + (amaxX_-aminX_)*d_randGen[tIdx+(s_randIdx++)*Nparticle_];
    pso_vel[q2_] = aminY_ + (amaxY_-aminY_)*d_randGen[tIdx+(s_randIdx++)*Nparticle_];
    pso_vel[q3_] = aminZ_ + (amaxZ_-aminZ_)*d_randGen[tIdx+(s_randIdx++)*Nparticle_];

    /* END INIT
     * pso_pos[Ndim] actual particle pose
     * pso_vel[Ndim] actual particle lin. and ang. velocity
     * pso_pos_b[Ndim] actul particle personal best pose
     */

    //copy back to global memory
    const uint32_t idx_tx = tIdx + tx_*Nparticle_;
    const uint32_t idx_ty = tIdx + ty_*Nparticle_;
    const uint32_t idx_tz = tIdx + tz_*Nparticle_;
    const uint32_t idx_q0 = tIdx + q0_*Nparticle_;
    const uint32_t idx_q1 = tIdx + q1_*Nparticle_;
    const uint32_t idx_q2 = tIdx + q2_*Nparticle_;
    const uint32_t idx_q3 = tIdx + q3_*Nparticle_;

//  printf("[ %f %f %f %f %f %f %f ]\n",pso_pos[tx_],pso_pos[ty_],pso_pos[tz_],
//          pso_pos[q0_],pso_pos[q1_],pso_pos[q2_],pso_pos[q3_]);

    /*Particles current Pose*/
    d_pso_pos[idx_tx] = pso_pos[tx_];
    d_pso_pos[idx_ty] = pso_pos[ty_];
    d_pso_pos[idx_tz] = pso_pos[tz_];

    d_pso_pos[idx_q0] = pso_pos[q0_];
    d_pso_pos[idx_q1] = pso_pos[q1_];
    d_pso_pos[idx_q2] = pso_pos[q2_];
    d_pso_pos[idx_q3] = pso_pos[q3_];
    /*Particles current Vel*/
    d_pso_vel[idx_tx] = pso_vel[tx_];
    d_pso_vel[idx_ty] = pso_vel[ty_];
    d_pso_vel[idx_tz] = pso_vel[tz_];

    d_pso_vel[idx_q0] = pso_vel[q0_];
    d_pso_vel[idx_q1] = pso_vel[q1_];
    d_pso_vel[idx_q2] = pso_vel[q2_];
    d_pso_vel[idx_q3] = pso_vel[q3_];
    /*Particles Personal Best */
    d_pso_pos_b[idx_tx] = pso_pos_b[tx_];
    d_pso_pos_b[idx_ty] = pso_pos_b[ty_];
    d_pso_pos_b[idx_tz] = pso_pos_b[tz_];

    d_pso_pos_b[idx_q0] = pso_pos_b[q0_];
    d_pso_pos_b[idx_q1] = pso_pos_b[q1_];
    d_pso_pos_b[idx_q2] = pso_pos_b[q2_];
    d_pso_pos_b[idx_q3] = pso_pos_b[q3_];

    //keep track of the indices
    d_randIdx[tIdx] = s_randIdx;


    return;
}
/*
__global__ void RenderFixedPointKernel(
                               //float* d_obj_model, //we use the texture instead !!
                               int* d_depth_buffer,
                               const uint32_t offset_buffer,
                               float* d_pso_pos,
                               const uint32_t pIdx)
{

    //one thread per triangle (face)
    uint32_t tIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t numFaces = d_qpso_params->numFaces;

    if(tIdx >= numFaces)
        return;
    //hardcoded pso_pose since we are testing the time of
    //one particle render
    //const float pso_pose_vec[7] = {0.0285172f,-0.16441f,0.815619f,
    //                              1.f, 0.0f, 0.f, 0.f};
//       From LoadOBJ3:
//       * we assumes one thread per triangle (Face) and we store
//       * the obj_model as [x0f0 x0f1 x0f2...x0fN |
//       *                  x1f0 x1f1 x1f2...x1fN  |
//       *                  x2f0 x2f1 x2f2...x2fN  | numVerts
//       *                  y0f0 y0f1 y0f2...y0fN  |
//       *                  y1f0 y1f1 y1f2...y1fN  |
//       *                  y2f0 y2f1 y2f2...y2fN  | 2*numVerts
//       *                  z0f0 z0f1 z0f2...z0fN  |
//       *                  z1f0 z1f1 z1f2...z1fN  |
//       *                  z2f0 z2f1 z2f2...z2fN  ] 2*numVerts
//       *
//       * where x0 y0 z0 is the first vertex of the face 0 (f0)...or face 58 (f58)
//       *

    //TODO: COMMENTED BY GIORGIO 2: aggiunto: float* d_pso_pos,
        //const uint32_t pIdx come parametri del kernel. Ogni particle ora legge la sua posa!

        //const float pso_pose_vec[7] = {0.0285172f,-0.16441f,0.815619f,
        //                              1.f, 0.0f, 0.f, 0.f};

        const uint32_t idx_tx = pIdx + tx_*Nparticle_;
        const uint32_t idx_ty = pIdx + ty_*Nparticle_;
        const uint32_t idx_tz = pIdx + tz_*Nparticle_;
        const uint32_t idx_q0 = pIdx + q0_*Nparticle_;
        const uint32_t idx_q1 = pIdx + q1_*Nparticle_;
        const uint32_t idx_q2 = pIdx + q2_*Nparticle_;
        const uint32_t idx_q3 = pIdx + q3_*Nparticle_;

        const float pso_pose_vec[7]={
            d_pso_pos[idx_tx],
            d_pso_pos[idx_ty],
            d_pso_pos[idx_tz],
            d_pso_pos[idx_q0],
            d_pso_pos[idx_q1],
            d_pso_pos[idx_q2],
            d_pso_pos[idx_q3]
        };

        // END COMMENTED BY GIORGIO 2


//  const uint32_t _3numFaces = 3*numFaces;
//  const uint32_t _6numFaces = 6*numFaces;

    float4 triangle[3];
    //float triangle[9]; // 3 vertices * 3 dims(x,y,z)


    #pragma unroll
    for(int j=0;j<3;++j)//three vertices of the face
    {

//
//      triangle[j].x = d_obj_model[tIdx + j*numFaces];
//      triangle[j].y = d_obj_model[tIdx + _3numFaces + j*numFaces];
//      triangle[j].z = d_obj_model[tIdx + _6numFaces + j*numFaces];
//
        //triangle[j] = tex1Dfetch(g_ord_obj_float4Texture,3*tIdx+j);
        triangle[j] = tex1Dfetch(g_ord_obj_float4Texture,numFaces*j+tIdx);

    }


    float4 tranformed_triangle[3];
    //first vertex of the face tIdx
    projectModelPointsToPixels(&pso_pose_vec[0],triangle[0],tranformed_triangle[0]);
    //second vertex of the face tIdx
    projectModelPointsToPixels(&pso_pose_vec[0],triangle[1],tranformed_triangle[1]);
    //third vertex of the face tIdx
    projectModelPointsToPixels(&pso_pose_vec[0],triangle[2],tranformed_triangle[2]);

    //Hereafter the Z component of each vertex is in millimeter and float

    //Each triangle (tIdx) is rendered in parallel
    renderFixedPoint(   tranformed_triangle[0],
                        tranformed_triangle[1],
                        tranformed_triangle[2],
                        d_depth_buffer,
                        offset_buffer,
                        tIdx);

}

*/


__global__ void RenderFixedPointKernelAABB(
                               float* d_obj_model, //we use the texture instead !!
                               int* d_depth_buffer,
                               const uint32_t offset_buffer,
                               float* d_pso_pos,
                               const uint32_t pIdx,
                               int32_t* d_AABB)
{

    //one thread per triangle (face)
    uint32_t tIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t numFaces = d_qpso_params->numFaces;

    //Each of max size of blockDim.x    (Faces per block)   //init
    volatile __shared__ int32_t sMinX[fxptThsPerBlock]; sMinX[threadIdx.x] = max_AABB_;
    volatile __shared__ int32_t sMaxX[fxptThsPerBlock]; sMaxX[threadIdx.x] = min_AABB_;
    volatile __shared__ int32_t sMinY[fxptThsPerBlock]; sMinY[threadIdx.x] = max_AABB_;
    volatile __shared__ int32_t sMaxY[fxptThsPerBlock]; sMaxY[threadIdx.x] = min_AABB_;
    __syncthreads();

    if(tIdx >= numFaces)
        return;
    //hardcoded pso_pose since we are testing the time of
    //one particle render
    //const float pso_pose_vec[7] = {0.0285172f,-0.16441f,0.815619f,
    //                              1.f, 0.0f, 0.f, 0.f};
    /*   From LoadOBJ3:
         * we assumes one thread per triangle (Face) and we store
         * the obj_model as [x0f0 x0f1 x0f2...x0fN |
         *                  x1f0 x1f1 x1f2...x1fN  |
         *                  x2f0 x2f1 x2f2...x2fN  | numVerts
         *                  y0f0 y0f1 y0f2...y0fN  |
         *                  y1f0 y1f1 y1f2...y1fN  |
         *                  y2f0 y2f1 y2f2...y2fN  | 2*numVerts
         *                  z0f0 z0f1 z0f2...z0fN  |
         *                  z1f0 z1f1 z1f2...z1fN  |
         *                  z2f0 z2f1 z2f2...z2fN  ] 2*numVerts
         *
         * where x0 y0 z0 is the first vertex of the face 0 (f0)...or face 58 (f58)
         */

    //TODO: COMMENTED BY GIORGIO 2: aggiunto: float* d_pso_pos,
        //const uint32_t pIdx come parametri del kernel. Ogni particle ora legge la sua posa!

//      const float pso_pose_vec[7] = {0.0285172f,-0.16441f,0.815619f,
//                                      1.f, 0.0f, 0.f, 0.f};

        const uint32_t idx_tx = pIdx + tx_*Nparticle_;
        const uint32_t idx_ty = pIdx + ty_*Nparticle_;
        const uint32_t idx_tz = pIdx + tz_*Nparticle_;
        const uint32_t idx_q0 = pIdx + q0_*Nparticle_;
        const uint32_t idx_q1 = pIdx + q1_*Nparticle_;
        const uint32_t idx_q2 = pIdx + q2_*Nparticle_;
        const uint32_t idx_q3 = pIdx + q3_*Nparticle_;

        const float pso_pose_vec[7]={
            d_pso_pos[idx_tx],
            d_pso_pos[idx_ty],
            d_pso_pos[idx_tz],
            d_pso_pos[idx_q0],
            d_pso_pos[idx_q1],
            d_pso_pos[idx_q2],
            d_pso_pos[idx_q3]
        };

        // END COMMENTED BY GIORGIO 2


    const uint32_t _3numFaces = 3*numFaces;
    const uint32_t _6numFaces = 6*numFaces;

    float4 triangle[3];
    //float triangle[9]; // 3 vertices * 3 dims(x,y,z)


    #pragma unroll
    for(int j=0;j<3;++j)//three vertices of the face
    {


        triangle[j].x = d_obj_model[tIdx + j*numFaces];
        triangle[j].y = d_obj_model[tIdx + _3numFaces + j*numFaces];
        triangle[j].z = d_obj_model[tIdx + _6numFaces + j*numFaces];

        //triangle[j] = tex1Dfetch(g_ord_obj_float4Texture,3*tIdx+j);
        /***triangle[j] = tex1Dfetch(g_ord_obj_float4Texture,numFaces*j+tIdx);***/

    }


    float4 tranformed_triangle[3];
    //first vertex of the face tIdx
    projectModelPointsToPixels(&pso_pose_vec[0],triangle[0],tranformed_triangle[0]);
    //second vertex of the face tIdx
    projectModelPointsToPixels(&pso_pose_vec[0],triangle[1],tranformed_triangle[1]);
    //third vertex of the face tIdx
    projectModelPointsToPixels(&pso_pose_vec[0],triangle[2],tranformed_triangle[2]);

    /*Hereafter the Z component of each vertex is in millimeter and float*/

    //Each triangle (tIdx) is rendered in parallel
    renderFixedPointAABB(tranformed_triangle[0],
                        tranformed_triangle[1],
                        tranformed_triangle[2],
                        d_depth_buffer,
                        offset_buffer,
                        tIdx,
                        pIdx,
                        sMinX,
                        sMaxX,
                        sMinY,
                        sMaxY,
                        d_AABB);

}



__global__ void ComputeAABBFinalReductionKernel(  //int* d_depth_buffer,
                                   //const uint32_t offset_buffer,
                                   //const uint32_t pIdx,
                                   int32_t* d_AABB,
                                   int32_t* d_finalAABB)
{


    /*** this kernel is launched with 1 block per particle and each block has 256 ths==8warps ***/


    const uint32_t tIdx = blockDim.x * blockIdx.x + threadIdx.x;

    const uint32_t globalWarpIdx = tIdx/32;
    const uint32_t blockWarpIdx = threadIdx.x/32;

    const uint32_t laneIdx = tIdx & 0x1F; //[0-31]
    const uint32_t blocklaneIdx = threadIdx.x & 0x1F; //[0-31]

    const uint32_t pIdx = blockIdx.x;


    /** Reconstruct the actual AABB of the rendered object pIdx **/
    /*
     * A particle needs 24 values (iff fxptBlocks==6)  to construct the AABB,
     * i.e. fxptBlocks minX's (6)
     *      fxptBlocks maxX's (6)
     *      fxptBlocks minY's (6)
     *      fxptBlocks maxY's (6) == 6*4 == 24
     * For each min/max we fill the remaining (64-6==)58 values with MAX or MIN in order to have
     * coalesced access to global memory.
     * TODO: reduce d_AABB to warpSize*4*Nparticle and fill the last warpSize values of the
     *  shared memories to min/max inside the kernel
     * TODO: better idea: use shuffle intrinsic fcns with CC >=3.0 and a warpSize only
     */
    volatile __shared__ int32_t sMinX[_2warpSize_];
    volatile __shared__ int32_t sMaxX[_2warpSize_];
    volatile __shared__ int32_t sMinY[_2warpSize_];
    volatile __shared__ int32_t sMaxY[_2warpSize_];

    //Usage of 8 warps== 32*8 == 256 threads
    if(blockWarpIdx<2)//chosen warp 0 and 1 to handle sMinX
    {
        uint32_t actualLaneIdx = blocklaneIdx+blockWarpIdx*_warpSize_; //[0-63]
        sMinX[actualLaneIdx] = d_AABB[pIdx*_8warpSize_  + actualLaneIdx];

    }
    if(blockWarpIdx>=2 && blockWarpIdx<4)//chosen warp 2 and 3 to handle sMaxX
    {
        uint32_t actualLaneIdx = blocklaneIdx+blockWarpIdx*_warpSize_; //[64-127]
        const uint32_t idx = blocklaneIdx+((blockWarpIdx-1)/2)*_warpSize_;
        sMaxX[idx] = d_AABB[pIdx*_8warpSize_  + actualLaneIdx];

    }
    if(blockWarpIdx>=4 && blockWarpIdx<6)//chosen warp 4 and 5 to handle sMinY
    {
        uint32_t actualLaneIdx = blocklaneIdx+blockWarpIdx*_warpSize_; //[128-191]
        const uint32_t idx = blocklaneIdx+((blockWarpIdx-3)/2)*_warpSize_;
        sMinY[idx] = d_AABB[pIdx*_8warpSize_  + actualLaneIdx];

    }
    if(blockWarpIdx>=6 && blockWarpIdx<8)//chosen warp 6 and 7 to handle sMaxY
    {
        uint32_t actualLaneIdx = blocklaneIdx+blockWarpIdx*_warpSize_; //[192-255]
        const uint32_t idx = blocklaneIdx+((blockWarpIdx-5)/2)*_warpSize_;
        sMaxY[idx] = d_AABB[pIdx*_8warpSize_  + actualLaneIdx];

    }
    __syncthreads();

    //Let's reduce...

    if(blockWarpIdx==0)//chosen warp 0 to handle sMinX
    {
        sMinX[blocklaneIdx] = min(sMinX[blocklaneIdx],sMinX[blocklaneIdx+32]);
        sMinX[blocklaneIdx] = min(sMinX[blocklaneIdx],sMinX[blocklaneIdx+16]);
        sMinX[blocklaneIdx] = min(sMinX[blocklaneIdx],sMinX[blocklaneIdx+8]);
        sMinX[blocklaneIdx] = min(sMinX[blocklaneIdx],sMinX[blocklaneIdx+4]);
        sMinX[blocklaneIdx] = min(sMinX[blocklaneIdx],sMinX[blocklaneIdx+2]);
        sMinX[blocklaneIdx] = min(sMinX[blocklaneIdx],sMinX[blocklaneIdx+1]);
    }
    if(blockWarpIdx==1)//chosen warp 1 to handle sMaxX
    {
        sMaxX[blocklaneIdx] = max(sMaxX[blocklaneIdx],sMaxX[blocklaneIdx+32]);
        sMaxX[blocklaneIdx] = max(sMaxX[blocklaneIdx],sMaxX[blocklaneIdx+16]);
        sMaxX[blocklaneIdx] = max(sMaxX[blocklaneIdx],sMaxX[blocklaneIdx+8]);
        sMaxX[blocklaneIdx] = max(sMaxX[blocklaneIdx],sMaxX[blocklaneIdx+4]);
        sMaxX[blocklaneIdx] = max(sMaxX[blocklaneIdx],sMaxX[blocklaneIdx+2]);
        sMaxX[blocklaneIdx] = max(sMaxX[blocklaneIdx],sMaxX[blocklaneIdx+1]);
    }
    if(blockWarpIdx==2)//chosen warp 2 to handle sMinY
    {
        sMinY[blocklaneIdx] = min(sMinY[blocklaneIdx],sMinY[blocklaneIdx+32]);
        sMinY[blocklaneIdx] = min(sMinY[blocklaneIdx],sMinY[blocklaneIdx+16]);
        sMinY[blocklaneIdx] = min(sMinY[blocklaneIdx],sMinY[blocklaneIdx+8]);
        sMinY[blocklaneIdx] = min(sMinY[blocklaneIdx],sMinY[blocklaneIdx+4]);
        sMinY[blocklaneIdx] = min(sMinY[blocklaneIdx],sMinY[blocklaneIdx+2]);
        sMinY[blocklaneIdx] = min(sMinY[blocklaneIdx],sMinY[blocklaneIdx+1]);
    }

    if(blockWarpIdx==3)//chosen warp 3 to handle sMinY
    {
        sMaxY[blocklaneIdx] = max(sMaxY[blocklaneIdx],sMaxY[blocklaneIdx+32]);
        sMaxY[blocklaneIdx] = max(sMaxY[blocklaneIdx],sMaxY[blocklaneIdx+16]);
        sMaxY[blocklaneIdx] = max(sMaxY[blocklaneIdx],sMaxY[blocklaneIdx+8]);
        sMaxY[blocklaneIdx] = max(sMaxY[blocklaneIdx],sMaxY[blocklaneIdx+4]);
        sMaxY[blocklaneIdx] = max(sMaxY[blocklaneIdx],sMaxY[blocklaneIdx+2]);
        sMaxY[blocklaneIdx] = max(sMaxY[blocklaneIdx],sMaxY[blocklaneIdx+1]);
    }

    //end reduce !

    //Each block so each particle has 4 values (an AABB)
    //store to global memory
    if(threadIdx.x==0)
    {
        d_finalAABB[pIdx*_warpSize_ + 0] = sMinX[0];
        d_finalAABB[pIdx*_warpSize_ + 1] = sMaxX[0];
        d_finalAABB[pIdx*_warpSize_ + 2] = sMinY[0];
        d_finalAABB[pIdx*_warpSize_ + 3] = sMaxY[0];
    }

    return;
}



__global__ void ComputeFitnessKernel(int* d_depth_buffer,
                               const uint32_t offset_buffer,
                               const uint32_t pIdx,
                               int32_t* d_finalAABB,
                               float* d_pso_fit_error,
                               int* d_depth_kinect)
{

    // This kernel is launched for each particle using streams

    //Let's assume 1 block of 1024 threads
    const uint32_t tIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const int32_t totalNumThreads = blockDim.x * gridDim.x;
//  const uint32_t blockWarpIdx = threadIdx.x/32;
//  const uint32_t blocklaneIdx = threadIdx.x & 0x1F; //[0-31]

    const unsigned int NClusterPoints = d_qpso_params->num_point_SegmentedCluster;

    //Get the AABB for this particle pIdx

    //TODO: TO AVOID BANK CONFLIT ????
    volatile __shared__ int32_t MinMaxXY[_warpSize_];

    volatile __shared__ float sError[2048];// 2*1024 threads
    //volatile __shared__ unsigned int sNrenderedPoints[2048];// 2*1024 threads
    volatile __shared__ unsigned int sNumSegDepthWoutValid3dModelPixel[2048];
    volatile __shared__ unsigned int sNumRenderedDepthWoutValid3dClusterPixel[2048];

    //Warp with 1024 threads? 1024/32 == 32 ==> 32*2 == 64
    volatile __shared__ unsigned int sNrenderedPointsWarpBallot[64];

    //init shared
    sError[tIdx]=0.f;sError[tIdx+1024]=0.f;
    sNumSegDepthWoutValid3dModelPixel[tIdx]=0;sNumSegDepthWoutValid3dModelPixel[tIdx+1024]=0;
    sNumRenderedDepthWoutValid3dClusterPixel[tIdx]=0;sNumRenderedDepthWoutValid3dClusterPixel[tIdx+1024]=0;
    //sNrenderedPoints[tIdx]=0;sNrenderedPoints[tIdx+1024]=0;

    if(tIdx<_warpSize_) // chosen first warp
    {
        MinMaxXY[tIdx] = d_finalAABB[pIdx*_warpSize_ + tIdx];
        sNrenderedPointsWarpBallot[tIdx] = 0; sNrenderedPointsWarpBallot[tIdx+32] = 0;
    }
    __syncthreads();


    //HERE BANK CONFLIT !!
    const int32_t minXr=MinMaxXY[0];
    const int32_t maxXr=MinMaxXY[1];
    const int32_t minYr=MinMaxXY[2];
    const int32_t maxYr=MinMaxXY[3];

    //TODO: HARDCODED AABB of the segmented cluster
     int32_t minXs = d_qpso_params[0].AABBminX;
     int32_t maxXs = d_qpso_params[0].AABBmaxX;
     int32_t minYs = d_qpso_params[0].AABBminY;
     int32_t maxYs = d_qpso_params[0].AABBmaxY;




    //OR of the 2 AABB
    const int32_t minX = min(minXr,minXs);
    const int32_t maxX = max(maxXr,maxXs);
    const int32_t minY = min(minYr,minYs);
    const int32_t maxY = max(maxYr,maxYs);


//  const int32_t minX=d_finalAABB[pIdx*_warpSize_ ];
//  const int32_t maxX=d_finalAABB[pIdx*_warpSize_ + 1 ];
//  const int32_t minY=d_finalAABB[pIdx*_warpSize_ + 2];
//  const int32_t maxY=d_finalAABB[pIdx*_warpSize_ + 3];


    const int32_t W_bar = maxX-minX;//width AABB maxX-minX
    const int32_t H_bar = maxY-minY;//Height AABB maxY-minY
    const int32_t Area = W_bar*H_bar;
    const float _1totalNumThreadsFloat  = __frcp_rn( __int2float_rn(totalNumThreads));
    //in Round-Up or round toward positive infinity cast == MATLAB ceil()
    const int32_t NumIters = __float2int_ru( Area*_1totalNumThreadsFloat );

    //printf("NumIters: %d ---> %f \n",NumIters,Area*_1totalNumThreadsFloat);

    for(int32_t xx=0; xx<NumIters; ++xx)
    {
        int32_t tIdxLoop = tIdx + xx*totalNumThreads;

        //unsigned int tIdxLoopU = (unsigned int)tIdxLoop;
        //unsigned int W_bar_U = (unsigned int)W_bar;
        //TODO: A faster remainder ?? look up tAble ?!
        //fixedpoint_modulo() returns sometimes wrong results (approximation errors)
        const int32_t x_bar = tIdxLoop % W_bar;//double_modulo(tIdxLoopU,W_bar_U);//fixedpoint_modulo(tIdxLoopU,W_bar_U);

        //printf("tIdxLoop: %d; x_bar FPmod: %d || x_bar mod: %d\n",tIdxLoop, x_bar,(tIdxLoop % W_bar));

        //Round-Towards-Zero mode == MATLAB fix()
        const int32_t y_bar = __float2int_rz(tIdxLoop * __frcp_rn( __int2float_rn(W_bar)) );

        //printf("y_bar: %d ---> %f \n",y_bar,tIdxLoop * __frcp_rn( __int2float_rn(W_bar)));

        const int32_t pxX = minX + x_bar; //minX + ...
        const int32_t pxY = minY + y_bar; //minY + ...

/*
        if(tIdx==0)
        {
            printf("tIdx: %d. tIdxLoop: %d. pxX: %d. pxY: %d. NumIters: %d. "
                    "minX: %d. maxX: %d. minY: %d. maxY: %d\n",tIdx,tIdxLoop,pxX,pxY,NumIters,minX,maxX,minY,maxY);
        }
        if(tIdx==1)
        {
                printf("tIdx: %d. tIdxLoop: %d. pxX: %d. pxY: %d. NumIters: %d. "
                        "minX: %d. maxX: %d. minY: %d. maxY: %d\n",tIdx,tIdxLoop,pxX,pxY,NumIters,minX,maxX,minY,maxY);
        }
        if(tIdx==2)
        {
                printf("tIdx: %d. tIdxLoop: %d. pxX: %d. pxY: %d. NumIters: %d. "
                        "minX: %d. maxX: %d. minY: %d. maxY: %d\n",tIdx,tIdxLoop,pxX,pxY,NumIters,minX,maxX,minY,maxY);
        }
*/

        if( pxX>=minX && pxY>=minY && pxX<=maxX && pxY<=maxY
             /*&& pxX<cols_ && pxY<rows_ && pxX>=0 && pxY>=0*/)
        {

            const int32_t pxIdxRendered = pxY*cols_ + pxX + offset_buffer;
            int z_rendered = d_depth_buffer[pxIdxRendered];

            /*TODO: Zeros out the rendered buffer for the next iteration.
             * MayBe can be done by another kernel, in parallel (using streams) with a
             * low occupancy one (e.g., UpdatePersonalAndGlobalBest)
             */
            d_depth_buffer[pxIdxRendered] = maxVal_depthBufferInt;



//          if(z_rendered==maxVal_depthBufferInt)
//              continue;

            const int32_t pxIdxKinect = pxY*cols_ + pxX;
            int z_kinect = d_depth_kinect[pxIdxKinect];//tex1Dfetch(textureDepthKinect,pxIdxKinect);

            // Circular array  0-2047 : 2048 is a power of 2 so the modulus turns into x & (y-1)
            // tIdxLoopCirc goes from [0 to 2047] and the shared memory is thus completely filled !
            int32_t tIdxLoopCirc = tIdxLoop & 2047;

            /**RATIO INDEX**/ //TODO: it does NOT work fine based on rendered AABB !!
            if(z_kinect/*>0*/ && z_rendered==maxVal_depthBufferInt)
            {
                sNumSegDepthWoutValid3dModelPixel[tIdxLoopCirc] += 1 ;
            }


//          if(!z_kinect)
//              continue;

            if(z_rendered!=maxVal_depthBufferInt)
            {

                /**RATIO TRANSPOSED INDEX**/ //it works fine based on rendered AABB !!
                if(!z_kinect)//z_kinect==0
                {
                    sNumRenderedDepthWoutValid3dClusterPixel[tIdxLoopCirc] += 1;

                }
                else
                {
                    float diff = __int2float_rn(z_rendered-z_kinect)*tom_;
                    sError[tIdxLoopCirc] +=  (diff)*(diff)/**(!(z_rendered==maxVal_depthBufferInt))*/;
                    //sNrenderedPoints[tIdxLoopCirc] += 1;
                }

                /*** !! Se sono forte....questa tecnica e' da Advanced Users...CAzzo!! ***/
                unsigned int mask = __ballot(1);  // mask of active lanes
                int NrenderedPixelWithinWarp = __popc(mask);
                int leader = __ffs(mask) - 1;  // -1 for 0-based indexing
                unsigned int laneID = (tIdxLoop & 1023) & 31;
                unsigned int WarpID = (tIdxLoop & 1023) / 32;
                if(laneID==leader)//only the first "active" lane in the warp writes to shared
                    sNrenderedPointsWarpBallot[WarpID] += NrenderedPixelWithinWarp;
            }

        }//end check inside AABB

    }//end for NumIters
    //At the end of the for We have a shared memory of 2048 elements composed by the partial sums of error
    //inside the AABB.
    //Now we have to sum-reduce the shared memory to obtain the total error

    //FitnessReduction(sError, sNrenderedPoints, tIdx);
    FitnessReductionBallot(sError,sNrenderedPointsWarpBallot,
                            sNumSegDepthWoutValid3dModelPixel,
                            sNumRenderedDepthWoutValid3dClusterPixel,tIdx);
    //TODO: __syncthread NOT NEEDED ?
    //__syncthreads();

    //Hereafter, sError[0] holds the total Fitness error for that particle (pIdx)
    if(tIdx==0)//save the resulting fitness error to global array
    {
//      printf("sNrenderedPoints[0]: %d. ---> Area: %d. bool (<) ? %d.\n",
//              sNrenderedPoints[0], Area, (sNrenderedPoints[0]<Area));
//      printf("sNrenderedPoints[0]: %d. sNrenderedPointsWarpBallot[0]: %d.\n",sNrenderedPoints[0],sNrenderedPointsWarpBallot[0]);

//      d_pso_fit_error[pIdx] = sError[0]* __frcp_rn( __uint2float_rn(sNrenderedPoints[0]));

        float _1_sNrenderedPointsWarpBallot = __frcp_rn( __uint2float_rn(sNrenderedPointsWarpBallot[0]));

        float ratio = sNumSegDepthWoutValid3dModelPixel[0]*__frcp_rn( __uint2float_rn(NClusterPoints));
        float ratioT = sNumRenderedDepthWoutValid3dClusterPixel[0]*_1_sNrenderedPointsWarpBallot;
        float Zerror = sError[0]*_1_sNrenderedPointsWarpBallot;

        d_pso_fit_error[pIdx] = (ratio+ratioT)*0.05 /* *0.005f x duck*/+ Zerror;//final error: (ratio+ratioT)/(2*10) + Zerror;
    }

    return;
}



__global__ void UpdatePersonalAndGlobalBest(float* d_pso_fit_error,//actual fitness error per particle [1*Nparticle]
                                            float* d_pso_personal_best_fit,//Personal BEST fitness error per particle [1*Nparticle]
                                            float* d_pso_pos_nb,//global best [1xDim] if global topology
                                            float* d_pso_pos, //particles current pose [Ndim*Nparticle]
                                            float* d_pso_vel, //particles current vels [Ndim*Nparticle]
                                            float* d_pso_pos_b, //particles best pose  [Ndim*Nparticle])
                                            float* d_solution_best_fit, //solution best fitness error [1x1]
                                            float* d_solution_best_pose, //solution best pose [1xNdim]
                                            float* d_result_min_fit,
                                            int best_particle_Idx)
{


    /* the kernel is launched as 8x128 == 1024 particle: 1 thread per particle*/
    const uint32_t tIdx = blockDim.x * blockIdx.x + threadIdx.x;
    //volatile __shared__ float stemp_global_best[128];//[1024];
    //stemp_global_best[threadIdx.x]=0.f;

    /** UPDATE PERSONAL UPDATE **/
    const float personal_actual_fit = d_pso_fit_error[tIdx];
    const float personal_best_fit = d_pso_personal_best_fit[tIdx];

    if(personal_actual_fit < personal_best_fit)
    {
        //update personal best fit
        d_pso_personal_best_fit[tIdx] = personal_actual_fit;
        //update particle best pose
        const uint32_t idx_tx = tIdx + tx_*Nparticle_;
        const uint32_t idx_ty = tIdx + ty_*Nparticle_;
        const uint32_t idx_tz = tIdx + tz_*Nparticle_;
        const uint32_t idx_q0 = tIdx + q0_*Nparticle_;
        const uint32_t idx_q1 = tIdx + q1_*Nparticle_;
        const uint32_t idx_q2 = tIdx + q2_*Nparticle_;
        const uint32_t idx_q3 = tIdx + q3_*Nparticle_;
        d_pso_pos_b[idx_tx] = d_pso_pos[idx_tx];
        d_pso_pos_b[idx_ty] = d_pso_pos[idx_ty];
        d_pso_pos_b[idx_tz] = d_pso_pos[idx_tz];
        d_pso_pos_b[idx_q0] = d_pso_pos[idx_q0];
        d_pso_pos_b[idx_q1] = d_pso_pos[idx_q1];
        d_pso_pos_b[idx_q2] = d_pso_pos[idx_q2];
        d_pso_pos_b[idx_q3] = d_pso_pos[idx_q3];

    }

    /** UPDATE GLOBAL BEST **/
    //float solution_best_fit = d_solution_best_fit[0];


//  if(personal_actual_fit<solution_best_fit)
//  {
//      //stemp_global_best[threadIdx.x] = personal_actual_fit;
//      atomicExch(&d_solution_best_fit[0],personal_actual_fit);
//  }


    if(tIdx==0)
    {
        const float min_fit = *d_result_min_fit;
        const int pIdx = best_particle_Idx;
        const float solution_best_fit = *d_solution_best_fit;
        //printf("Kernel:: min_fit: %f. d_solution_best_fit[0]: %f.\n",min_fit,solution_best_fit);
        if(min_fit < solution_best_fit)
        {
            //update global best
            *d_solution_best_fit = min_fit;
            //update the global best pose
            const uint32_t idx_tx = pIdx + tx_*Nparticle_;
            const uint32_t idx_ty = pIdx + ty_*Nparticle_;
            const uint32_t idx_tz = pIdx + tz_*Nparticle_;
            const uint32_t idx_q0 = pIdx + q0_*Nparticle_;
            const uint32_t idx_q1 = pIdx + q1_*Nparticle_;
            const uint32_t idx_q2 = pIdx + q2_*Nparticle_;
            const uint32_t idx_q3 = pIdx + q3_*Nparticle_;
            d_solution_best_pose[tx_] = d_pso_pos[idx_tx];
            d_solution_best_pose[ty_] = d_pso_pos[idx_ty];
            d_solution_best_pose[tz_] = d_pso_pos[idx_tz];
            d_solution_best_pose[q0_] = d_pso_pos[idx_q0];
            d_solution_best_pose[q1_] = d_pso_pos[idx_q1];
            d_solution_best_pose[q2_] = d_pso_pos[idx_q2];
            d_solution_best_pose[q3_] = d_pso_pos[idx_q3];

        }
    }



    return;
}


__global__ void UpdateParticlesEquationKernel(float* d_solution_best_pose,//float* d_pso_pos_nb,//global best pose 1xDim if global topology
                                            float* d_randGen,
                                            float* d_pso_pos, //particles current pose [Ndim*Nparticle]
                                            float* d_pso_vel, //particles current vels [Ndim*Nparticle]
                                            float* d_pso_pos_b, //particles best pose  [Ndim*Nparticle]
                                            unsigned int* d_randIdx) //to keep track of the index each thread has within the d_randGen vector

{

    /** This kernel is launched as 8x128 == 1024 == 1 thread x particle **/
    const uint32_t tIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t idx_tx = tIdx + tx_*Nparticle_;
    const uint32_t idx_ty = tIdx + ty_*Nparticle_;
    const uint32_t idx_tz = tIdx + tz_*Nparticle_;
    const uint32_t idx_q0 = tIdx + q0_*Nparticle_;
    const uint32_t idx_q1 = tIdx + q1_*Nparticle_;
    const uint32_t idx_q2 = tIdx + q2_*Nparticle_;
    const uint32_t idx_q3 = tIdx + q3_*Nparticle_;

    float pso_vel[Ndim_];
    float pso_pos[Ndim_];
    float pso_pos_b[Ndim_];
    float pso_pos_nb[Ndim_];

    /*Particles current Pose*/
    pso_pos[tx_] = d_pso_pos[idx_tx];
    pso_pos[ty_] = d_pso_pos[idx_ty];
    pso_pos[tz_] = d_pso_pos[idx_tz];

    pso_pos[q0_] = d_pso_pos[idx_q0];
    pso_pos[q1_] = d_pso_pos[idx_q1];
    pso_pos[q2_] = d_pso_pos[idx_q2];
    pso_pos[q3_] = d_pso_pos[idx_q3];
    /*Particles current Vel*/
    pso_vel[tx_] = d_pso_vel[idx_tx];
    pso_vel[ty_] = d_pso_vel[idx_ty];
    pso_vel[tz_] = d_pso_vel[idx_tz];

    pso_vel[q0_] = d_pso_vel[idx_q0];
    pso_vel[q1_] = d_pso_vel[idx_q1];
    pso_vel[q2_] = d_pso_vel[idx_q2];
    pso_vel[q3_] = d_pso_vel[idx_q3];
    /*Particles Personal Best */
    pso_pos_b[tx_] = d_pso_pos_b[idx_tx];
    pso_pos_b[ty_] = d_pso_pos_b[idx_ty];
    pso_pos_b[tz_] = d_pso_pos_b[idx_tz];

    pso_pos_b[q0_] = d_pso_pos_b[idx_q0];
    pso_pos_b[q1_] = d_pso_pos_b[idx_q1];
    pso_pos_b[q2_] = d_pso_pos_b[idx_q2];
    pso_pos_b[q3_] = d_pso_pos_b[idx_q3];

    /* GLOBAL BEST POSE */
    pso_pos_nb[tx_] = d_solution_best_pose[tx_];
    pso_pos_nb[ty_] = d_solution_best_pose[ty_];
    pso_pos_nb[tz_] = d_solution_best_pose[tz_];
    pso_pos_nb[q0_] = d_solution_best_pose[q0_];
    pso_pos_nb[q1_] = d_solution_best_pose[q1_];
    pso_pos_nb[q2_] = d_solution_best_pose[q2_];
    pso_pos_nb[q3_] = d_solution_best_pose[q3_];


    // Read Random vector Indices
    unsigned int s_randIdx = d_randIdx[tIdx];

    //const translation bounds:
    const float x_lo[3] = {d_qpso_params[0].x_lo[tx_],d_qpso_params[0].x_lo[ty_],d_qpso_params[0].x_lo[tz_]};
    const float x_hi[3] = {d_qpso_params[0].x_hi[tx_],d_qpso_params[0].x_hi[ty_],d_qpso_params[0].x_hi[tz_]};
    /* POSITION UPDATE */
    float pso_rho1=0.f;
    float pso_rho2=0.f;

    #pragma unroll
    for (int _d_ = 0; _d_ < q0_; ++_d_) {

        //TODO: use c1_ c2_ from #define and not from __device__ constant memory
        pso_rho1 = c1_ * d_randGen[tIdx+(s_randIdx++)*Nparticle_];
        pso_rho2 = c2_ * d_randGen[tIdx+(s_randIdx++)*Nparticle_];
        //TODO: Until Now use PSO weight from #define and not from __device__ global memory
        //LINERA VELOCITY eq. tx_ , ty_ , tz_
        float iV = psoW_*pso_vel[_d_];
        float cV = pso_rho1*(pso_pos_b[_d_] - pso_pos[_d_]);
        float sV = pso_rho2*(pso_pos_nb[_d_] - pso_pos[_d_]);
        pso_vel[_d_] =  iV + cV + sV;

        //Update Position
        pso_pos[_d_] += pso_vel[_d_];

        // clamp position within bounds?
        if (pso_pos[_d_] < x_lo[_d_]) {
            pso_pos[_d_] = x_lo[_d_];
            pso_vel[_d_] = 0.f;
        } else if (pso_pos[_d_] > x_hi[_d_]) {
            pso_pos[_d_] = x_hi[_d_];
            pso_vel[_d_] = 0.f;
        }


    }//end for tx_ ty_ tz_ (LINEAR PART)

    /*QUATERNION UPDATE */
    float q_tilde_cognitive[4];
    float q_tilde_social[4];

    /** COGNITIVE PART **/

    //Quaternion Product q_ind_best*q_actual^(-1)
    QuatProd_pq_1PSOstateVect(pso_pos_b,pso_pos,q_tilde_cognitive);
    Cast2TopHalfHypersphere(q_tilde_cognitive);

    const float norm_v_tilde_cognitive = sqrtf(//__fsqrt_rn(//sqrt(
                    q_tilde_cognitive[1]*q_tilde_cognitive[1] +
                    q_tilde_cognitive[2]*q_tilde_cognitive[2] +
                    q_tilde_cognitive[3]*q_tilde_cognitive[3]
                                         );
    float w_tilde_cognitive[3]={0.0f,0.0f,0.0f};
    //double scaleW = 1;

    if (norm_v_tilde_cognitive>MIN_NORM_)
    {
        const float _1overNorm_v_tilde_cognitive = __frcp_rn(norm_v_tilde_cognitive);
        const float _2acos_q_tilde_cognitive_overNorm = 2.0f*acosf(q_tilde_cognitive[0]) * _1overNorm_v_tilde_cognitive;

        w_tilde_cognitive[0] = q_tilde_cognitive[1] * _2acos_q_tilde_cognitive_overNorm;

        w_tilde_cognitive[1] = q_tilde_cognitive[2] * _2acos_q_tilde_cognitive_overNorm;

        w_tilde_cognitive[2] = q_tilde_cognitive[3] * _2acos_q_tilde_cognitive_overNorm;
    }

    /** SOCIAL PART **/

    //Quaternion Product q_global_best*q_actual^(-1)
    QuatProd_pq_1PSOstateVect(pso_pos_nb,pso_pos,q_tilde_social);
    Cast2TopHalfHypersphere(q_tilde_social);

    const float norm_v_tilde_social = sqrtf(
                 q_tilde_social[1]*q_tilde_social[1] +
                 q_tilde_social[2]*q_tilde_social[2] +
                 q_tilde_social[3]*q_tilde_social[3]
                                       );

    float w_tilde_social[3]={0.0f,0.0f,0.0f};
    if (norm_v_tilde_social>MIN_NORM_)
    {
        const float _1overNorm_v_tilde_social = __frcp_rn(norm_v_tilde_social);
        const float _2acos_q_tilde_social_overNorm = 2.0f*acosf(q_tilde_social[0]) * _1overNorm_v_tilde_social;

        w_tilde_social[0] = q_tilde_social[1] * _2acos_q_tilde_social_overNorm;

        w_tilde_social[1] = q_tilde_social[2] * _2acos_q_tilde_social_overNorm;

        w_tilde_social[2] = q_tilde_social[3] * _2acos_q_tilde_social_overNorm;
    }
    /**UPDATE the Angular Velocity**/
    pso_vel[q0_] = 0.0f;

    #pragma unroll
    for (int w_idx=1; w_idx<4; ++w_idx) {

        // calculate stochastic coefficients
        pso_rho1 = c1_ * d_randGen[tIdx+(s_randIdx++)*Nparticle_]; //((*UniDisPtr)(*genPtr));
        pso_rho2 = c2_ * d_randGen[tIdx+(s_randIdx++)*Nparticle_];

        const float iW = psoW_ * pso_vel[w_idx+q0_];
        const float cW = pso_rho1*w_tilde_cognitive[w_idx-1];
        const float sW = pso_rho2*w_tilde_social[w_idx-1];

        pso_vel[w_idx+q0_] = iW + cW + sW;

    }

    /**UPDATE the Quaternions**/
    QuatKinematicsPSOStateVect(pso_pos,pso_vel);


    //Update the Rand indices
    d_randIdx[tIdx] = s_randIdx;

    //Save Back the new:
    /*Particles actual Pose*/
    d_pso_pos[idx_tx] = pso_pos[tx_];
    d_pso_pos[idx_ty] = pso_pos[ty_];
    d_pso_pos[idx_tz] = pso_pos[tz_];

    d_pso_pos[idx_q0] = pso_pos[q0_];
    d_pso_pos[idx_q1] = pso_pos[q1_];
    d_pso_pos[idx_q2] = pso_pos[q2_];
    d_pso_pos[idx_q3] = pso_pos[q3_];

    /*Particles actual Vel*/
    d_pso_vel[idx_tx] = pso_vel[tx_];
    d_pso_vel[idx_ty] = pso_vel[ty_];
    d_pso_vel[idx_tz] = pso_vel[tz_];

    d_pso_vel[idx_q0] = pso_vel[q0_];
    d_pso_vel[idx_q1] = pso_vel[q1_];
    d_pso_vel[idx_q2] = pso_vel[q2_];
    d_pso_vel[idx_q3] = pso_vel[q3_];


    return;
}



__global__ void ComputeFitnessKernelMultiBlock(int* d_depth_buffer,
                               const uint32_t offset_buffer,
                               const uint32_t pIdx,
                               int32_t* d_finalAABB,
                               float* d_pso_fit_error,
                               int* d_depth_kinect)
{

    // This kernel is launched for each particle using streams

    //Let's assume 8 blocks of 512 threads == 4096 threads
    const uint32_t tIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const int32_t totalNumThreads = blockDim.x * gridDim.x;

    //Get the AABB for this particle pIdx
    //int32_t MinMaxXY[_warpSize_];

    //TODO: TO AVOID BANK CONFLIT ????
//  volatile __shared__ int32_t MinMaxXY[_warpSize_];

    volatile __shared__ float sError[1024];//Each block has 2*512 dim of shared memory (512==threads x block)
    //init shared
    sError[threadIdx.x]=0.f;sError[threadIdx.x+512]=0.f;

//  if(tIdx<_warpSize_) // chosen first warp
//  {
//      MinMaxXY[tIdx] = d_finalAABB[pIdx*_warpSize_ + tIdx];
//  }
//  __syncthreads();


    //HERE BANK CONFLIT !!
//  const int32_t minX=MinMaxXY[0];
//  const int32_t maxX=MinMaxXY[1];
//  const int32_t minY=MinMaxXY[2];
//  const int32_t maxY=MinMaxXY[3];

    const int32_t minX=d_finalAABB[pIdx*_warpSize_ ];
    const int32_t maxX=d_finalAABB[pIdx*_warpSize_ + 1 ];
    const int32_t minY=d_finalAABB[pIdx*_warpSize_ + 2];
    const int32_t maxY=d_finalAABB[pIdx*_warpSize_ + 3];


    const int32_t W_bar = maxX-minX;//width AABB maxX-minX
    const int32_t H_bar = maxY-minY;//width AABB maxY-minY
    const int32_t Area = W_bar*H_bar;
    const float _1totalNumThreadsFloat  = __frcp_rn( __int2float_rn(totalNumThreads));
    //in Round-Up or round toward positive infinity cast == MATLAB ceil()
    const int32_t NumIters = __float2int_ru( Area*_1totalNumThreadsFloat );

    for(int32_t xx=0; xx<NumIters; ++xx)
    {
        int32_t tIdxLoop = tIdx + xx*totalNumThreads;

        unsigned int tIdxLoopU = (unsigned int)tIdxLoop;
        unsigned int W_bar_U = (unsigned int)W_bar;
        //TODO: A faster remainder ?? look up tAble ?!
        const int32_t x_bar = fixedpoint_modulo(tIdxLoopU,W_bar_U);//tIdxLoop % W_bar;

        //printf("tIdxLoop: %d; x_bar FPmod: %d || x_bar mod: %d\n",tIdxLoop, x_bar,(tIdxLoop % W_bar));
        //printf("tIdxLoop: %d; x_bar mod: %d\n",tIdxLoop, (tIdxLoop % W_bar));

        //Round-Towards-Zero mode == MATLAB fix()
        const int32_t y_bar = __float2int_rz(tIdxLoop * __frcp_rn( __int2float_rn(W_bar)) );

        const int32_t pxX = minX + x_bar; //minX + ...
        const int32_t pxY = minY + y_bar; //minY + ...

        if(/*pxX>=minX && pxY>=minY &&*/ pxX<=maxX && pxY<=maxY
             /*&& pxX<cols_ && pxY<rows_ && pxX>=0 && pxY>=0*/)
        {

            const int32_t pxIdxRendered = pxY*cols_ + pxX + offset_buffer;
            int z_rendered = d_depth_buffer[pxIdxRendered];

            const int32_t pxIdxKinect = pxY*cols_ + pxX;

            int z_kinect = d_depth_kinect[pxIdxKinect];

            //TODO: circular array ?? 0-1023 : 1024 is a power of 2 so the modulus turns into x & (y-1)
            // tIdxLoopCirc goes from [0 to 1023] and the shared memory is thus completely filled !
            int32_t tIdxLoopCirc = tIdxLoop & 1023;//2047;
            //int32_t diff = z_rendered-z_kinect;
            float diff = __int2float_rn(z_rendered-z_kinect)*tom_;
            //if z_rendered is NaN, zeros out the error !
            //TODO:RACE CONDITION?? NEED __synchtreads() ???
            if(z_rendered!=maxVal_depthBufferInt)
                sError[tIdxLoopCirc] +=  (diff)*(diff)/**(!(z_rendered==maxVal_depthBufferInt))*/;

        }//end check inside AABB

    }//end for NumIters
    //At the end of the for We have a shared memory of 1023 elements composed by the partial sums of error
    //inside the AABB, FOR EACH BLOCK !!!
    //Now we have to sum-reduce the shared memory to obtain the total error.

    FitnessReductionMultiBlock(sError, tIdx);

    //Hereafter, sError[0] of each block (blockIdx.x) holds the partial Fitness error for that particle (pIdx)

    if(tIdx==0)//save the resulting fitness error to global array
    {
        //TODO: we are HURRY !!! we save only the result of the first block !!
        //TODO: we need a total sum-reduction over all the blocks !!!
        if(blockIdx.x==0)
            d_pso_fit_error[pIdx] = sError[0];
    }

    return;
}


__global__ void FillTheDepthBufferMaxValKernel(int* d_depth_buffer)
{

    /* We assume one thread per pixel of an image RxC_
     * so RxC_ thread. A for loop of Nparticle_ fill all the depth
     * buffer of dimension RxC_*Nparticle_
     */
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int tIdx = y*cols_ + x;
    #pragma unroll
    for(uint32_t idx=0; idx<Nparticle_; ++idx)
    {
        //const int  bidx = tIdx + RxC_*idx;
        d_depth_buffer[ tIdx + RxC_*idx ] = maxVal_depthBufferInt;//value;
    }

    return;
}



int main(int argc, char **argv) {

	if(argc != 5)
	{
		printf("Usage: %s path/to/.obj color_image depth_image cluster_idx\n",argv[0]);
		return 0;
	}
	testAsyncStreamAvailable();

    //segmentation
    cv::Mat SegResRGB;
    cv::Mat SegResDepthM;
    cv::Mat SegResDepthRGB;

    std::string IDcode = "10";//argv[3];

    float kfloat = (float)k/10000.f;
    float kxfloat = (float)kx/1000.f;
    float kyfloat = (float)ky/1000.f;
    float ksfloat = (float)ks/1000.f;
    float gafloat = ((float)g_angle)*deg2rad;
    float lafloat = ((float)l_angle)*deg2rad;
    float lcannyf = (float)Lcanny/1000.f;
    float hcannyf = (float)Hcanny/1000.f;

    std::string rgb_file_path = argv[2]; //"img_1164.png";
    std::string depth_file_path = argv[3]; //"img_1164_depth.png";
    std::string rgb_name= argv[2];
    std::string depth_name= argv[3];    
//std::string obj_name= "Shampoo_small";
    //std::string obj_name= "Coffee_Cup_small";
    //std::string obj_name= "Juice_Carton_small";
    std::string obj_name="Joystick_small";
    rgb_file_path = "/home/morpheus/benchviseblue/data/"+rgb_name;
    depth_file_path = "/home/morpheus/benchviseblue/data/"+depth_name;
    
std::cout << "rgb_file: " << rgb_file_path << " depth_file: " << depth_file_path<<"\n"; 
    cv::Mat imageMask = cv::Mat();
    // rgb_name= "img_134.png";
    // obj_name= "Shampoo";
    // rgb_file_path = "/Volumes/HD-PNTU3/datasets/"+obj_name+"/RGB/"+rgb_name;
    // depth_file_path = "/Volumes/HD-PNTU3/datasets/"+obj_name+"/Depth/"+rgb_name;
    kinect_rgb_img = cv::imread(rgb_file_path,cv::IMREAD_UNCHANGED);
    kinect_depth_img_mm = loadDepth(depth_file_path);//,cv::IMREAD_UNCHANGED);// in mm
    //cv::imshow("kinect_rgb_img",kinect_rgb_img);
    //cv::Mat testDepthRead;
    //GraphCanny::GraphCannySeg<GraphCanny::hsv>::convertDepth2ColorMap(kinect_depth_img_mm,testDepthRead);
    //cv::imshow("testDepth",testDepthRead);    
//cv::waitKey(0);




    GraphCanny::GraphCannySeg<GraphCanny::hsv>* gcs;
    std::vector<GraphCanny::SegResults> vecSegResult;
    gcs = new GraphCanny::GraphCannySeg<GraphCanny::hsv>(kinect_rgb_img, kinect_depth_img_mm, sigma, kfloat, min_size, kxfloat, kyfloat, ksfloat,k_vec,lcannyf,hcannyf,kdv, kdc,max_ecc,max_L1,max_L2,(uint16_t)DTH,(uint16_t)plusD,(uint16_t)point3D,gafloat,lafloat,(float)FarObjZ);
    gcs->run();

    vecSegResult = gcs->vecSegResults;
    for(int i=0;i<vecSegResult.size();++i)   // show results
    {
    //     cv::Rect& rect_ = vecSegResult[i].rect_aabb_;
    //     cv::rectangle(vecSegResult[i].clusterRGB,rect_, cv::Scalar(255));
    //     printf("Rect H: %d, W: %d, Area: %d, tl: [%d , %d], br: [%d , %d] \n",rect_.height,rect_.width,rect_.area(), rect_.tl().x, rect_.tl().y, rect_.br().x, rect_.br().y);
	char s[100];sprintf(s,"results/%s-cluster%d.jpg",rgb_name.c_str(),i);
         cv::imwrite(s, vecSegResult[i].clusterRGB);
    //     cv::waitKey(0);
     }


    // pso
    int IDX_cluster = atoi(argv[4]); // clutser id

    std::cout<<"centroid IDX_cluster: "<< IDX_cluster <<" "<<vecSegResult[IDX_cluster].centroid3D<<"\n------\n";
    cv::imwrite("cluster.jpg",vecSegResult[IDX_cluster].clusterRGB);
    //cv::imwrite("cluster_depth.jpg",vecSegResult[IDX_cluster].clusterDepth);
    // for(int i=0;i<640*480;i++)
    //     if(vecSegResult[IDX_cluster].clusterDepth.data[i]>0) printf("%d ",(uint16_t)vecSegResult[IDX_cluster].clusterDepth.data[i]);
    cv::Rect aabb=vecSegResult[IDX_cluster].rect_aabb_;
    // min xy, max xy
    //float4 myBB = make_float4(aabb.tl().x,aabb.tl().y,aabb.br().x,aabb.br().y);
    //printf("myBB: %f %f %f %f\n",myBB.x,myBB.y,myBB.z,myBB.w);

    //visualize segmentation results
    //cv::imshow("CRGB", vecSegResult[IDX_cluster].clusterRGB);
    //GraphCanny::GraphCannySeg<GraphCanny::hsv>::visualizeColorMap(vecSegResult[IDX_cluster].clusterDepth,"W1",5);

    float segCentroidX=vecSegResult[IDX_cluster].centroid3D.x*0.001f;
    float segCentroidY=vecSegResult[IDX_cluster].centroid3D.y*0.001f;
    float segCentroidZ=vecSegResult[IDX_cluster].centroid3D.z*0.001f;
    //int *savedmatMM=(int *)vecSegResult[IDX_cluster].clusterDepth.data;
    //uint16_t *p=kinect_depth_img_mm.ptr<uint16_t>(0);
    //int *savedmatMM=(int *)malloc(RxC_*sizeof(int));
    //for(int i=0;i<RxC_;i++)
      //  if(vecSegResult[IDX_cluster].clusterDepth.data[i]==0)
        //    savedmatMM[i]=0;
        //else
          //  savedmatMM[i]=(int)p[i];
        //printf("\nCluster depth:\n");
    //for(int i=0;i<RxC_;i++)
        //if(savedmatMM[i]>0) printf("%d ",savedmatMM[i]);//savedmatMM[i]);


	std::string path_to_obj(argv[1]);

	QPSOParams_t* temp_qpso_params = (QPSOParams_t*)malloc(sizeof(QPSOParams_t));
	/** Init params **/
	temp_qpso_params->c1 = 1.f;
	temp_qpso_params->c2 = 1.f;
temp_qpso_params->AABBminX = aabb.tl().x;
temp_qpso_params->AABBminY = aabb.tl().y;
temp_qpso_params->AABBmaxX = aabb.br().x;
temp_qpso_params->AABBmaxY = aabb.br().y;
	//TODO: RUN the Segmentation and get the Cluster IDx and Centroid
	//FAKE computation of the Segmented Cluster Points
	//unsigned int size_Segmented_cluster=0;
	//for (int i = 0; i < RxC_; ++i) {

	//	if(savedmatMM[i] > 0)
	//	{
	//		++size_Segmented_cluster;
	//	}
	//}
	printf("Segmented Cluster Points: %lu \n",vecSegResult[IDX_cluster].num_points);
	temp_qpso_params->num_point_SegmentedCluster=(int)vecSegResult[IDX_cluster].num_points;
	//in [meters]
	float t_bound = 0.3;//0.04f;//0.04f; //in m
	float t_bound_z = 0.3f;//0.1 //in m
	temp_qpso_params->x_lo[0] = segCentroidX-t_bound;//-0.00948835f; //min tx
	temp_qpso_params->x_hi[0] = segCentroidX+t_bound;//0.0705116f; //max tx
	temp_qpso_params->x_lo[1] = segCentroidY-t_bound;//-0.213609f; //min ty
	temp_qpso_params->x_hi[1] = segCentroidY+t_bound;//-0.133609; //max ty
	temp_qpso_params->x_lo[2] = segCentroidZ-t_bound_z;//0.700522f; //min tz
	temp_qpso_params->x_hi[2] = segCentroidZ+t_bound_z;//0.900522f; //max tz



	printf("End Init QPSOParams_t\n");

    //Generate the RandomNumberVector GNU
    float* h_randGen = (float*)calloc(dimRandGen_,sizeof(float));
    float* d_randGen;
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    // seed the generator //Comment this to get the same value
    //gsl_rng_set(r, time(0));
    for (int i = 0; i < dimRandGen_; ++i)
    {
       h_randGen[i] = static_cast<float>(gsl_rng_uniform (r));
      //printf ("%.5f\n", h_randGen[i]);
    }
    gsl_rng_free (r);
    printf("End Init Random Vector\n");
    //return 0;
    //OrderedOBJVertices_t ord_obj;

    int numVerts=0;
    int numFaces=0;
    //FOR TEXTURE MEMORY
//  std::vector<float4> ord_obj_float4;
//  loadOBJ5(path_to_obj,ord_obj_float4,numVerts,numFaces);


    /*OBJ MODEL*/
    //with std::vector<float>
    std::vector<float> ord_obj;//[xxxxx,yyyyy,zzzzz] == size 3*numVerts;
    loadOBJ3(path_to_obj,ord_obj,numVerts,numFaces);
    //save the vertices number in constant memory
    temp_qpso_params->numVerts = numVerts;
    temp_qpso_params->numFaces = numFaces;
    float* d_obj_model;
    HANDLE_ERROR(cudaMalloc((void**)&d_obj_model, 3*numVerts*sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_obj_model, ord_obj.data(), 3*numVerts*sizeof(float), cudaMemcpyHostToDevice));
    printf("Object Model Loaded to device\n");


    /* Rand ARRAY */
    HANDLE_ERROR(cudaMalloc((void **)&d_randGen, dimRandGen_*sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_randGen, h_randGen, dimRandGen_*sizeof(float), cudaMemcpyHostToDevice));
    //to keep track of the index of each particle inside the d_randGen vector
    unsigned int* h_randIdx = (unsigned int*)calloc(Nparticle_,sizeof(unsigned int));
    unsigned int* d_randIdx;
    HANDLE_ERROR(cudaMalloc((void **)&d_randIdx, Nparticle_*sizeof(unsigned int)));
    HANDLE_ERROR(cudaMemcpy(d_randIdx, h_randIdx, Nparticle_*sizeof(unsigned int), cudaMemcpyHostToDevice));
    //free(h_randIdx);

    printf("Random ARRAY Loaded to device\n");

    /* PSO GLOBAL BEST POSE : in pso global topology this is 1xDim */
    float *d_pso_pos_nb = 0;
    float *h_pso_pos_nb = (float*)calloc(Ndim_,sizeof(float));
    HANDLE_ERROR(cudaMalloc((void **)&d_pso_pos_nb, Ndim_*sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_pso_pos_nb, h_pso_pos_nb, Ndim_*sizeof(float), cudaMemcpyHostToDevice));
    printf("PSO GLOBAL BEST POSE Loaded to device\n");

    /* PSO SOLUTION BEST FIT*/
    float* d_solution_best_fit = 0;
    float* h_solution_best_fit = (float*)malloc(sizeof(float));
    *h_solution_best_fit = max_best_fit_;
    HANDLE_ERROR(cudaMalloc((void **)&d_solution_best_fit,sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_solution_best_fit, h_solution_best_fit, sizeof(float), cudaMemcpyHostToDevice));
    //free(h_solution_best_fit);
    printf("PSO SOLUTION BEST FIT Loaded to device\n");

    /* PSO SOLUTION BEST POSE*/
    float* d_solution_best_pose = 0;
    float* h_solution_best_pose = (float*)calloc(Ndim_,sizeof(float));
    HANDLE_ERROR(cudaMalloc((void **)&d_solution_best_pose,Ndim_*sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_solution_best_pose, h_solution_best_pose, Ndim_*sizeof(float), cudaMemcpyHostToDevice));
    //free(h_solution_best_pose);
    printf("PSO SOLUTION BEST POSE Loaded to device\n");

    /*
     * pso_pos[Ndim*Nparticle_] actual particle pose
     * pso_vel[Ndim*Nparticle_] actual particle lin. and ang. velocity
     * pso_pos_b[Ndim*Nparticle_] actual particle personal best pose
    */
     float* h_pso_pos = (float*)calloc(Ndim_*Nparticle_,sizeof(float));
     float* d_pso_pos = 0;
     HANDLE_ERROR(cudaMalloc((void **)&d_pso_pos,Ndim_*Nparticle_*sizeof(float)));
     HANDLE_ERROR(cudaMemcpy(d_pso_pos, h_pso_pos, Ndim_*Nparticle_*sizeof(float), cudaMemcpyHostToDevice));
     float* h_pso_vel = (float*)calloc(Ndim_*Nparticle_,sizeof(float));
     float* d_pso_vel = 0;
     HANDLE_ERROR(cudaMalloc((void **)&d_pso_vel,Ndim_*Nparticle_*sizeof(float)));
     HANDLE_ERROR(cudaMemcpy(d_pso_vel, h_pso_vel, Ndim_*Nparticle_*sizeof(float), cudaMemcpyHostToDevice));
     float* h_pso_pos_b = (float*)calloc(Ndim_*Nparticle_,sizeof(float));
     float* d_pso_pos_b = 0;
     HANDLE_ERROR(cudaMalloc((void **)&d_pso_pos_b,Ndim_*Nparticle_*sizeof(float)));
     HANDLE_ERROR(cudaMemcpy(d_pso_pos_b, h_pso_pos_b, Ndim_*Nparticle_*sizeof(float), cudaMemcpyHostToDevice));
//   free(h_pso_pos);

//   free(h_pso_pos_b);

    /* Inertia Weight */
    float* d_psoW = NULL;
    float* h_psoW = (float*)calloc(1,sizeof(float));
    HANDLE_ERROR(cudaMalloc((void **)&d_psoW,sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_psoW, h_psoW, sizeof(float), cudaMemcpyHostToDevice));

    printf("Inertia Weight Loaded to device\n");

    /** CONSTANT PARAMS **/
    HANDLE_ERROR( cudaMemcpyToSymbol( d_qpso_params, temp_qpso_params,sizeof(QPSOParams_t)) );

    printf("CONSTANT PARAMS Loaded to device\n");


   /** FITNESS ERROR FOR EACH PARTICLE **/
   float* d_pso_fit_error;
   float* h_pso_fit_error = (float*)calloc(Nparticle_,sizeof(float));
   HANDLE_ERROR(cudaMalloc((void **)&d_pso_fit_error,Nparticle_*sizeof(float)));
   HANDLE_ERROR(cudaMemcpy(d_pso_fit_error, h_pso_fit_error, Nparticle_*sizeof(float), cudaMemcpyHostToDevice));
   //free(h_pso_fit_error);
   printf("FITNESS ERROR FOR EACH PARTICLE Loaded to device\n");

   /** PERSONAL BEST FITNESS ERROR FOR EACH PARTICLE **/
  float* d_pso_personal_best_fit;
  float* h_pso_personal_best_fit = (float*)calloc(Nparticle_,sizeof(float));
  //INIT to MAX
  for(int i=0;i<Nparticle_;++i)
  {
      h_pso_personal_best_fit[i] = max_best_fit_;
  }
  HANDLE_ERROR(cudaMalloc((void **)&d_pso_personal_best_fit,Nparticle_*sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_pso_personal_best_fit, h_pso_personal_best_fit, Nparticle_*sizeof(float), cudaMemcpyHostToDevice));

  printf("PERSONAL BEST FITNESS ERROR FOR EACH PARTICLE Loaded to device\n");



    /* RENDERED DEPTH IMAGE */
    printf("start calloc render\n");
    int* h_depth_buffer = (int*)calloc(RenderedDepthBufferSize_,sizeof(int));
    if(!h_depth_buffer)
    {
        printf("ERROR calloc render\n");
        return 0;
    }
    //fill the host depth buffer with max value
//  float* maxPtr = h_depth_buffer + RenderedDepthBufferSize_;
//  float* ptr = h_depth_buffer;
//  while(ptr < maxPtr)
//      *ptr++ = 60000.0f;

    int* d_depth_buffer;
    HANDLE_ERROR(cudaMalloc((void **)&d_depth_buffer,RenderedDepthBufferSize_*sizeof(int)));
    //printf("between cudaMalloc and ducaMemcpy\n");
    HANDLE_ERROR(cudaMemcpy(d_depth_buffer, h_depth_buffer, RenderedDepthBufferSize_*sizeof(int), cudaMemcpyHostToDevice));
    printf("RENDERED DEPTH IMAGE Loaded to device\n");
    //free(h_depth_buffer);

    /*CONSTANT OBJ 1024faces*/
//  HANDLE_ERROR(cudaMemcpyToSymbol(shampoo1024const, sh1024faces, 1024*9*sizeof(float)) );


    /* To build the AABB for each rendered obj*/
    /*
     * Each block save its minX minY maxX maxY in global d_AABB
     * To be processed by another kernel to get the final AABB of the obj
     *
     */

    size_t AABB_buffer_size = Nparticle_*_2warpSize_*4;//Nparticle_*fxptBlocks*4;
    int32_t* h_AABB = (int32_t*)calloc(AABB_buffer_size,sizeof(int32_t));

    //fill the host h_AABB array
    for(uint32_t pIdx=0;pIdx<Nparticle_;++pIdx)
    {
        for(uint32_t lane=0;lane<_2warpSize_;++lane)
        {
            uint32_t idx0 = pIdx*(_8warpSize_) + lane;
            uint32_t idx64 = idx0+64;
            uint32_t idx128 = idx0+128;
            uint32_t idx192 = idx0+192;

            h_AABB[idx0] = max_AABB_; //minX
            h_AABB[idx64] = min_AABB_;//maxX
            h_AABB[idx128] = max_AABB_;//minY
            h_AABB[idx192] = min_AABB_;//maxY

        }
    }

    int32_t* d_AABB;
    HANDLE_ERROR(cudaMalloc((void **)&d_AABB,AABB_buffer_size*sizeof(int32_t)));
    HANDLE_ERROR(cudaMemcpy(d_AABB, h_AABB, AABB_buffer_size*sizeof(int32_t), cudaMemcpyHostToDevice));



    /*  Final AABB */
    size_t finalAABB_buffer_size = Nparticle_*_warpSize_;//Nparticle_*fxptBlocks*4;
    int32_t* h_finalAABB = (int32_t*)calloc(finalAABB_buffer_size,sizeof(int32_t));
    int32_t* d_finalAABB;
    HANDLE_ERROR(cudaMalloc((void **)&d_finalAABB,finalAABB_buffer_size*sizeof(int32_t)));
    HANDLE_ERROR(cudaMemcpy(d_finalAABB, h_finalAABB, finalAABB_buffer_size*sizeof(int32_t), cudaMemcpyHostToDevice));
    //free(h_finalAABB);

    /* TEXTURE INIT*/
    /*
    float4* d_vertices__;
    HANDLE_ERROR( cudaMalloc((void**)&d_vertices__, numVerts*sizeof(float4)) );
    HANDLE_ERROR( cudaMemcpy(d_vertices__, ord_obj_float4.data(), numVerts*sizeof(float4), cudaMemcpyHostToDevice) );
    cudaChannelFormatDesc channel4desc = cudaCreateChannelDesc<float4>();
    HANDLE_ERROR( cudaBindTexture(NULL, &g_ord_obj_float4Texture, d_vertices__, &channel4desc, numVerts*sizeof(float4)) );
     */

    /* TEXTURE DEPTH KINECT CLUSTER */
    int *h_DepthCluster=(int *)malloc(RxC_*sizeof(int));

    uint16_t* clusterDepthPtr = vecSegResult[IDX_cluster].clusterDepth.ptr<uint16_t>(0);
    for(int idd=0;idd<RxC_;++idd)
    {
    		h_DepthCluster[idd] = static_cast<int>(clusterDepthPtr[idd]);
    }
    int* d_depth_kinect;
    HANDLE_ERROR( cudaMalloc((void**)&d_depth_kinect, RxC_*sizeof(int)) );
    HANDLE_ERROR( cudaMemcpy(d_depth_kinect, h_DepthCluster, RxC_*sizeof(int), cudaMemcpyHostToDevice) );
    //cudaChannelFormatDesc channel4desc = cudaCreateChannelDesc<int>();
    //HANDLE_ERROR( cudaBindTexture(NULL, &textureDepthKinect, d_depth_kinect, &channel4desc, RxC_*sizeof(int)) );

    /*****************************
     **                          **
     ** START THE QPSO ALGORITHM **
     **                              **
     ******************************/

/***    ***** INITIALIZE the STREAMS ****
 * *
 * * */
    cudaStream_t *streams = (cudaStream_t *) malloc(NumStreams * sizeof(cudaStream_t));
    for (int i = 0; i < NumStreams; ++i)
    {
        HANDLE_ERROR(cudaStreamCreate(&(streams[i])));
    }
    Timer myTimer;
    dim3 thperblock(8,8);//64 ths per block
    //grid of 80*60 blocks
    //8*8*80*60 = RxC_ == 640x480 == 307200
    dim3 numBlock(cols_/thperblock.x,rows_/thperblock.y);
/*******
* *
* * **** END INITIALIZE the STREAMS *****/

/********* FILL TWICE THE DEPTH BUFFER WITH MAX VAL ****
* *
* * */
    myTimer.start();
    FillTheDepthBufferMaxValKernel<<<numBlock,thperblock>>>(d_depth_buffer/*,60000.0f*/);
    HANDLE_ERROR(cudaDeviceSynchronize());
    myTimer.stop();
    myTimer.getElapsedTime(Timer::MILLI,"FillTheDepthBufferMaxValKernel (~1.2GB per ~314.6Mpixels!!) elapsedTime");
    //AGAIN TO WARM-UP
    myTimer.start();
    FillTheDepthBufferMaxValKernel<<<numBlock,thperblock>>>(d_depth_buffer/*,60000.0f*/);
    HANDLE_ERROR(cudaDeviceSynchronize());
    myTimer.stop();
    myTimer.getElapsedTime(Timer::MILLI,"FillTheDepthBufferMaxValKernel AGAIN to WARM-UP (~1.2GB per ~314.6Mpixels!!) elapsedTime");
/********
* *
* * ***** END FILL TWICE THE DEPTH BUFFER WITH MAX VAL *****/

/********* LAUNCH KERNEL INIT PSO ****
* *
* * */
    myTimer.start();
    //
    InitQPSO<<<8,128>>>(d_pso_pos_nb,d_randGen,d_pso_pos,d_pso_vel,d_pso_pos_b,d_randIdx);
    HANDLE_ERROR(cudaDeviceSynchronize());
    myTimer.stop();
    myTimer.getElapsedTime(Timer::MILLI,"Init PSO elapsedTime");

/********
* *
* * ***** END LAUNCH KERNEL INIT PSO *****/


#define RUN_PSO
#ifdef RUN_PSO

/****** START THE TIMER ******/
    myTimer.start();

/****** LOOP with the Number of Iterations (STEPs) ******/
    #pragma unroll
    for(int num_steps_idx=0;num_steps_idx<max_steps_;++num_steps_idx)
    {
/********* LAUNCH KERNEL RenderFixedPointKernelAABB (RENDERING) ****
* *
* * */
        uint32_t offset_buffer=0;
        uint32_t particleID=0;
        #pragma unroll
        for(int idx=0;idx<ForEachParticle_;++idx)//for each particle
        {

            #pragma unroll
            for(int Sidx=0;Sidx<NumStreams;++Sidx)
            {
                RenderFixedPointKernelAABB<<<fxptBlocks,fxptThsPerBlock,0,streams[Sidx]>>>(d_obj_model,d_depth_buffer,offset_buffer,d_pso_pos,particleID,d_AABB);
                offset_buffer+=RxC_;
                ++particleID;
            }

        }
        HANDLE_ERROR(cudaDeviceSynchronize());
/********
* *
* * ***** END LAUNCH KERNEL RenderFixedPointKernelAABB (RENDERING) *****/


/********* LAUNCH KERNEL ComputeAABBFinalReductionKernel (FIND AABB per particle) ****
* *
* * */
        ComputeAABBFinalReductionKernel<<<Nparticle_,256>>>(d_AABB,d_finalAABB);
        HANDLE_ERROR(cudaDeviceSynchronize());
/********
* *
* * ***** END LAUNCH KERNEL ComputeAABBFinalReductionKernel (FIND AABB per particle) *****/



/********* LAUNCH KERNEL ComputeFitnessKernel (Compute Fitness per particle) ****
* *
* * */
        //zeros out
        offset_buffer=0;
        particleID=0;
        #pragma unroll
        for(int32_t idx=0;idx<ForEachParticle_;++idx)//for each particle
        {

            #pragma unroll
            for(int32_t Sidx=0;Sidx<NumStreams;++Sidx)
            {
                ComputeFitnessKernel<<<1,1024,0,streams[Sidx]>>>(d_depth_buffer,offset_buffer,particleID,d_finalAABB,d_pso_fit_error,d_depth_kinect);
                offset_buffer+=RxC_;
                ++particleID;
            }

        }
        HANDLE_ERROR(cudaDeviceSynchronize());
/********
* *
* * ***** END LAUNCH KERNEL ComputeFitnessKernel (Compute Fitness per particle) *****/


/********* LAUNCH KERNEL UpdatePersonalAndGlobalBest (UPDATE PERSONAL & GLOBAL BEST) ****
* *
* * */
        // wrap raw pointer with a device_ptr
        thrust::device_ptr<float> dev_ptr(d_pso_fit_error);
        thrust::device_ptr<float> result_min_fit = thrust::min_element(thrust::device, dev_ptr, dev_ptr + Nparticle_);
        //TODO: This is NOT the GLOBAL BEST PARTICLE.
        //TODO: It is the best Particle of this Iteration!!!!
        int best_particle_Idx = thrust::distance(dev_ptr,result_min_fit);

        // extract raw pointer from device_ptr
        float* d_result_min_fit = thrust::raw_pointer_cast(result_min_fit);


        UpdatePersonalAndGlobalBest<<<8,128>>>(d_pso_fit_error,d_pso_personal_best_fit,d_pso_pos_nb,
                d_pso_pos,d_pso_vel,d_pso_pos_b,d_solution_best_fit,d_solution_best_pose,
                d_result_min_fit,best_particle_Idx);
        HANDLE_ERROR(cudaDeviceSynchronize());
/********
* *
* * ***** END LAUNCH KERNEL UpdatePersonalAndGlobalBest (UPDATE PERSONAL & GLOBAL BEST) *****/

    // added by ste
    cudaMemcpy(h_solution_best_fit, d_solution_best_fit, 1*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Solution Best Fit: %f\n",*h_solution_best_fit);
    cudaMemcpy(h_solution_best_pose, d_solution_best_pose, Ndim_*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Solution Best Pose: ");
    printVector(h_solution_best_pose,Ndim_);


/********* LAUNCH KERNEL UpdateParticlesEquationKernel (UPDATE POSE & VELOCITY per particle) ****
* *
* * */
        UpdateParticlesEquationKernel<<<8,128>>>(d_solution_best_pose,d_randGen,d_pso_pos,d_pso_vel,d_pso_pos_b,d_randIdx);
        HANDLE_ERROR(cudaDeviceSynchronize());
/********
* *
* * ***** END LAUNCH KERNEL UpdateParticlesEquationKernel (UPDATE POSE & VELOCITY per particle) *****/


    }
/****** END LOOP with the Number of Iterations (STEPs) ******/

/****** STOP THE TIMER ******/
    myTimer.stop();
    printf("After %d QPSO iterations\n",max_steps_);
    myTimer.getElapsedTime(Timer::MILLI,"***QPSO Algorithm elapsedTime");

/****** READ BACK THE BEST POSE & BEST FITNESS *******/
    HANDLE_ERROR(cudaMemcpy(h_solution_best_fit, d_solution_best_fit, 1*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Solution Best Fit: %f\n",*h_solution_best_fit);

    HANDLE_ERROR(cudaMemcpy(h_solution_best_pose, d_solution_best_pose, Ndim_*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Solution Best Pose: ");
    printVector(h_solution_best_pose,Ndim_);


/***** SAVE THE BEST POSE RENDER without reading
     * the entire depthbuffer since it has been zeroed out
******/
    float pDepthBuffer[RxC_];
    #pragma unroll
    for (int i = 0; i < RxC_; ++i) {
        pDepthBuffer[i] = maxVal_depthBuffer;
    }
    char savepose_path[100];  
    sprintf(savepose_path,"results/%s-%d-bestpose.csv",rgb_name.c_str(),IDX_cluster);
    FILE *out=fopen(savepose_path,"w");
    fprintf(out,"%f,%f,%f,%f,%f,%f,%f,%f",*h_solution_best_fit,h_solution_best_pose[0],h_solution_best_pose[1],h_solution_best_pose[2],h_solution_best_pose[3],h_solution_best_pose[4],
        h_solution_best_pose[5],h_solution_best_pose[6]);
    fclose(out);
    std::string saveCVS_path= "results/"+rgb_name+"-"+ NumberToString(IDX_cluster)+"-best.csv";

    //std::string saveCVS_path = "best.csv";
    printf("Rendering in CPU the Best Pose in %s\n",saveCVS_path.c_str());
    RenderCPU(&pDepthBuffer[0],/*pose*/h_solution_best_pose,ord_obj,numFaces,saveCVS_path.c_str());
    //TODO: for better visualization use OpenGL with the best pose !!!


    HANDLE_ERROR(cudaMemcpy(h_pso_pos, d_pso_pos, Ndim_*Nparticle_*sizeof(float), cudaMemcpyDeviceToHost));

    //render all the particles
    #pragma unroll
    for(int pIdx=0;pIdx<10;++pIdx)
    {
        #pragma unroll
        for (int i = 0; i < RxC_; ++i) {
                pDepthBuffer[i] = maxVal_depthBuffer;
        }
        float pose[Ndim_]={0.f};
        saveCVS_path = NumberToString(pIdx) + ".csv";
        printOneParticlePoseFromGlobalMem(h_pso_pos,pIdx,pose);
        RenderCPU(&pDepthBuffer[0],pose,ord_obj,numFaces,saveCVS_path);
    }

#endif



#ifdef TEST_KERNELS


    //Read Back just in case...
    /*
    cudaMemcpy(h_randIdx, d_randIdx, Nparticle_*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("Indices of RandGen vector after Init PSO:\n");
    printVector(h_randIdx,Nparticle_);


    cudaMemcpy(h_pso_pos_b, d_pso_pos_b, Ndim_*Nparticle_*sizeof(float), cudaMemcpyDeviceToHost);
    for(int th=0;th<Nparticle_;++th)
    {
        for(int dim=0;dim<7;++dim)
        {
            printf("%f ",h_pso_pos_b[th + dim*Nparticle_]);
        }
        printf("\n");
    }
    */

//  free(h_randIdx);



    myTimer.start();
    uint32_t offset_buffer=0;
    //TODO: COMMENTED BY GIORGIO 3: aggiunti ...,d_pso_pos,particleID) come ultimi
    //due parametri del kernel, e l'incremento ++particleID;
    uint32_t particleID=0;
    /**** Launch kernel RenderFixedPointKernelAABB *****/
    #pragma unroll
    for(uint32_t idx=0;idx<ForEachParticle_/*128 256*/;++idx)//for each particle
    {

        #pragma unroll
        for(uint32_t Sidx=0;Sidx<NumStreams;++Sidx)
        {
            RenderFixedPointKernelAABB<<<fxptBlocks,fxptThsPerBlock,0,streams[Sidx]>>>(d_obj_model,d_depth_buffer,offset_buffer,d_pso_pos,particleID,d_AABB);
            offset_buffer+=RxC_;
            ++particleID;
        }

    }

    HANDLE_ERROR(cudaDeviceSynchronize());
    myTimer.stop();
    myTimer.getElapsedTime(Timer::MILLI,"RenderFixedPoint per 1024p: elapsedTime");


    myTimer.start();
    //launch kernel ComputeAABBFinalReductionKernel
    ComputeAABBFinalReductionKernel<<<Nparticle_,256>>>(d_AABB,d_finalAABB);
    HANDLE_ERROR(cudaDeviceSynchronize());
    myTimer.stop();
    myTimer.getElapsedTime(Timer::MILLI,"ComputeAABBFinalReductionKernel");
    //READ BACK final AABB per particle
    cudaMemcpy(h_finalAABB, d_finalAABB, finalAABB_buffer_size*sizeof(int32_t), cudaMemcpyDeviceToHost);
    //just print the AABB of the first particle (pIdx=0)
    const int32_t pIdx=0;
    int32_t minX = h_finalAABB[pIdx*_warpSize_];
    int32_t maxX = h_finalAABB[pIdx*_warpSize_+1];
    int32_t minY = h_finalAABB[pIdx*_warpSize_+2];
    int32_t maxY = h_finalAABB[pIdx*_warpSize_+3];
    printf("pIdx: %d :: minX: %d, maxX: %d, minY: %d, maxY: %d\n",pIdx,minX,maxX,minY,maxY);



    //zeros out
    offset_buffer=0;
    particleID=0;
    myTimer.start();
    /**** Launch kernel !!ComputeFitness!! *****/
    #pragma unroll
    for(uint32_t idx=0;idx<ForEachParticle_;++idx)//for each particle
    {

        #pragma unroll
        for(uint32_t Sidx=0;Sidx<NumStreams;++Sidx)
        {
            ComputeFitnessKernel<<<1,1024,0,streams[Sidx]>>>(d_depth_buffer,offset_buffer,particleID,d_finalAABB,d_pso_fit_error,d_depth_kinect,myBB);
            offset_buffer+=RxC_;
            ++particleID;
        }

    }
    HANDLE_ERROR(cudaDeviceSynchronize());
    myTimer.stop();
    myTimer.getElapsedTime(Timer::MILLI,"ComputeFitnessKernel");
    //READ BACK final AABB per particle
    cudaMemcpy(h_pso_fit_error, d_pso_fit_error, Nparticle_*sizeof(float), cudaMemcpyDeviceToHost);
//  for(int i=0;i<Nparticle_;++i)
//  {
//      printf("%f | ",h_pso_fit_error[i]);
//  }
//  printf("\n");




    /** UPDATE PERSONAL AND GLOBAL BEST **/

    myTimer.start();
    // wrap raw pointer with a device_ptr
    thrust::device_ptr<float> dev_ptr(d_pso_fit_error);
    thrust::device_ptr<float> result_min_fit = thrust::min_element(thrust::device, dev_ptr, dev_ptr + Nparticle_);
    //TODO: This is NOT the GLOBAL BEST PARTICLE.
    //It is the best Particle of this Iteration!!!!
    int best_particle_Idx = thrust::distance(dev_ptr,result_min_fit);

    // extract raw pointer from device_ptr
    float* d_result_min_fit = thrust::raw_pointer_cast(result_min_fit);
    //int* d_pos_result_min_fit;

    UpdatePersonalAndGlobalBest<<<8,128>>>(d_pso_fit_error,d_pso_personal_best_fit,d_pso_pos_nb,
            d_pso_pos,d_pso_vel,d_pso_pos_b,d_solution_best_fit,d_solution_best_pose,
            d_result_min_fit,best_particle_Idx);
    HANDLE_ERROR(cudaDeviceSynchronize());
    myTimer.stop();
    myTimer.getElapsedTime(Timer::MILLI,"UpdatePersonalAndGlobalBest");
    printf("Best Particle pIdx: %d\n",best_particle_Idx);
    /*READ BACK*/
    cudaMemcpy(h_solution_best_fit, d_solution_best_fit, 1*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Solution Best Fit: %f\n",*h_solution_best_fit);


    cudaMemcpy(h_solution_best_pose, d_solution_best_pose, Ndim_*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Solution Best Pose: ");
    printVector(h_solution_best_pose,Ndim_);
    //free(h_solution_best_pose);

    /* UPDATE PARTICLES EQ KERNEL */
    myTimer.start();
    UpdateParticlesEquationKernel<<<8,128>>>(d_solution_best_pose,d_randGen,d_pso_pos,d_pso_vel,d_pso_pos_b,d_randIdx);
    HANDLE_ERROR(cudaDeviceSynchronize());
    myTimer.stop();
    myTimer.getElapsedTime(Timer::MILLI,"UpdateParticlesEquationKernel");
    /*READ BACK RANDOM INDICES */ //TODO:24 used out of 50 per particle !!!
    cudaMemcpy(h_pso_pos, d_pso_pos, Ndim_*Nparticle_*sizeof(float), cudaMemcpyDeviceToHost);

    float* pose = printOneParticlePoseFromGlobalMem(h_pso_pos,best_particle_Idx);
    //printf("pose: ");
    //printVector(pose,Ndim_);
    //  cudaMemcpy(h_randIdx, d_randIdx, Nparticle_*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//  printf("Indices of RandGen vector after UpdateParticlesEquationKernel:\n");
//  printVector(h_randIdx,Nparticle_);



    /*Read Back the Depth Buffer*/
/*
    printf("Read Back the Depth Buffer...\n");
    cudaMemcpy(h_depth_buffer, d_depth_buffer, RenderedDepthBufferSize_*sizeof(int), cudaMemcpyDeviceToHost);

    //std::string BestParticlePath = static_cast<std::ostringstream*>( &(ostringstream() << best_particle_Idx) )->str();

    std::string BestParticleString = NumberToString(best_particle_Idx);

    std::string saveCVS_path = "/home/morpheus/myRayRender/best" + BestParticleString + ".csv";
    saveDepthRenderedCSVBestParticle(saveCVS_path,h_depth_buffer,best_particle_Idx);
    printf("Saved Best particle in [ %s ]\n",saveCVS_path.c_str());
*/

    //SAVE THE BEST POSE RENDER without downloading the entire depthbuffer since it is zeroed out
    float pDepthBuffer[RxC_];
    for (int i = 0; i < RxC_; ++i) {
        pDepthBuffer[i] = maxVal_depthBuffer;
    }
    //read back the best pose
    printf("Read Back the Best Pose...\n");
    HANDLE_ERROR(cudaMemcpy(h_solution_best_pose, d_solution_best_pose, Ndim_*sizeof(float), cudaMemcpyDeviceToHost));
    printf("h_solution_best_pose: ");
    printVector(h_solution_best_pose,Ndim_);
    std::string savepose_path = "results/"+rgb_name+"-"+IDX_cluster+"-bestpose.csv";
    FILE *out=fopen(savepose_path,"w");
    fprintf(out,"%f,%f,%f,%f,%f,%f",h_solution_best_pose[0],h_solution_best_pose[1],h_solution_best_pose[2],h_solution_best_pose[3],h_solution_best_pose[4],
	h_solution_best_pose[5],h_solution_best_pose[6],h_solution_best_pose[7]);
    fclose(out); 	
    std::string saveCSV_path = "results/"+rgb_name+"-"+IDX_cluster+"-best.csv";
    printf("Rendering in CPU the Best Pose in %s\n",saveCSV_path.c_str());
    RenderCPU(&pDepthBuffer[0],/*pose*/h_solution_best_pose,ord_obj,numFaces,saveCSV_path);





#endif




    //Free HOST
    free(h_pso_pos_nb);
    free(h_pso_vel);
    free(h_psoW);
    free(temp_qpso_params);
    free(h_pso_personal_best_fit);
    free(h_AABB);
    free(h_pso_pos_b);
    free(h_finalAABB);
    free(h_pso_fit_error);
    free(h_solution_best_fit);
    free(h_pso_pos);
    free(h_randIdx);
    free(h_depth_buffer);
    free(h_solution_best_pose);
    free(h_randGen);
    free(h_DepthCluster);


    //Free CUDA
    /*DESTROY STREAMS*/
    for (int i = 0; i < NumStreams; ++i)
    {
        cudaStreamDestroy(streams[i]);
        //cudaEventDestroy(kernelEvent[i]);
    }
    free(streams);
    cudaFree(d_pso_pos_nb);
    cudaFree(d_pso_pos_b);
    cudaFree(d_pso_pos);
    cudaFree(d_pso_vel);
    cudaFree(d_randGen);
    cudaFree(d_randIdx);
    cudaFree(d_solution_best_fit);
    cudaFree(d_psoW);
    cudaFree(d_obj_model);
    cudaFree(d_depth_buffer);
    cudaFree(d_AABB);
    cudaFree(d_finalAABB);
    cudaFree(d_pso_fit_error);
    cudaFree(d_solution_best_pose);
    cudaFree(d_pso_personal_best_fit);
    printf( "Daje.....Finite tutte' e cose cv::waitKey(0) press enter to quit !!\n");
cv::waitKey(0);

    return 0;
}

