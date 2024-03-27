#pragma once 

#include "Vector.h"
#include <cstring>

struct Matrix4f{
    __host__ __device__ Matrix4f() {
        m[0][0] = 1; m[0][1] = 0; m[0][2] = 0; m[0][3] = 0;
        m[1][0] = 0; m[1][1] = 1; m[1][2] = 0; m[1][3] = 0;
        m[2][0] = 0; m[2][1] = 0; m[2][2] = 1; m[2][3] = 0;
        m[3][0] = 0; m[3][1] = 0; m[3][2] = 0; m[3][3] = 1;
    }

    __host__ __device__ Matrix4f(float m00, float m01, float m02, float m03,
                                 float m10, float m11, float m12, float m13,
                                 float m20, float m21, float m22, float m23,
                                 float m30, float m31, float m32, float m33) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }

    __host__ __device__ Matrix4f(float v) {
        m[0][0] = v; m[0][1] = v; m[0][2] = v; m[0][3] = v;
        m[1][0] = v; m[1][1] = v; m[1][2] = v; m[1][3] = v;
        m[2][0] = v; m[2][1] = v; m[2][2] = v; m[2][3] = v;
        m[3][0] = v; m[3][1] = v; m[3][2] = v; m[3][3] = v;
    }

    __host__ __device__ Matrix4f(float mat[4][4]) {
        m[0][0] = mat[0][0]; m[0][1] = mat[0][1]; m[0][2] = mat[0][2]; m[0][3] = mat[0][3];
        m[1][0] = mat[1][0]; m[1][1] = mat[1][1]; m[1][2] = mat[1][2]; m[1][3] = mat[1][3];
        m[2][0] = mat[2][0]; m[2][1] = mat[2][1]; m[2][2] = mat[2][2]; m[2][3] = mat[2][3];
        m[3][0] = mat[3][0]; m[3][1] = mat[3][1]; m[3][2] = mat[3][2]; m[3][3] = mat[3][3];
    }

    __host__ __device__ bool operator==(const Matrix4f& mat) const {
        return m[0][0] == mat.m[0][0] && m[0][1] == mat.m[0][1] && m[0][2] == mat.m[0][2] && m[0][3] == mat.m[0][3] &&
               m[1][0] == mat.m[1][0] && m[1][1] == mat.m[1][1] && m[1][2] == mat.m[1][2] && m[1][3] == mat.m[1][3] &&
               m[2][0] == mat.m[2][0] && m[2][1] == mat.m[2][1] && m[2][2] == mat.m[2][2] && m[2][3] == mat.m[2][3] &&
               m[3][0] == mat.m[3][0] && m[3][1] == mat.m[3][1] && m[3][2] == mat.m[3][2] && m[3][3] == mat.m[3][3];
    }

    __host__ __device__ bool operator!=(const Matrix4f& mat) const {
        return m[0][0] != mat.m[0][0] || m[0][1] != mat.m[0][1] || m[0][2] != mat.m[0][2] || m[0][3] != mat.m[0][3] ||
               m[1][0] != mat.m[1][0] || m[1][1] != mat.m[1][1] || m[1][2] != mat.m[1][2] || m[1][3] != mat.m[1][3] ||
               m[2][0] != mat.m[2][0] || m[2][1] != mat.m[2][1] || m[2][2] != mat.m[2][2] || m[2][3] != mat.m[2][3] ||
               m[3][0] != mat.m[3][0] || m[3][1] != mat.m[3][1] || m[3][2] != mat.m[3][2] || m[3][3] != mat.m[3][3];
    }

    inline __host__ __device__ Matrix4f Transpose() const {
        return Matrix4f(m[0][0], m[1][0], m[2][0], m[3][0],
                        m[0][1], m[1][1], m[2][1], m[3][1],
                        m[0][2], m[1][2], m[2][2], m[3][2],
                        m[0][3], m[1][3], m[2][3], m[3][3]);
    }

    inline __host__ __device__ Matrix4f Inverse() const {
        int indxc[4], indxr[4];
        int ipiv[4] = {0, 0, 0, 0};
        float minv[4][4];
        memcpy(minv, m, 4*4*sizeof(float));
        for (int i = 0; i < 4; i++) {
            int irow = -1, icol = -1;
            float big = 0;
            // Choose pivot
            for (int j = 0; j < 4; j++) {
                if (ipiv[j] != 1) {
                    for (int k = 0; k < 4; k++) {
                        if (ipiv[k] == 0) {
                            if (fabsf(minv[j][k]) >= big) {
                                big = fabsf(minv[j][k]);
                                irow = j;
                                icol = k;
                            }
                        } else if (ipiv[k] > 1) {
                            printf("Singular matrix in MatrixInvert\n");
                        }
                    }
                }
            }
            ++ipiv[icol];
            // Swap rows irow and icol for pivot
            if (irow != icol) {
                for (int k = 0; k < 4; k++) {
                    float temp = minv[irow][k];
                    minv[irow][k] = minv[icol][k];
                    minv[icol][k] = temp;
                }
            }
            indxr[i] = irow;
            indxc[i] = icol;
            if (minv[icol][icol] == 0) {
                printf("Singular matrix in MatrixInvert\n");
            }
            // Set $m[icol][icol]$ to one by scaling row icol appropriately
            float pivinv = 1.f / minv[icol][icol];
            minv[icol][icol] = 1.f;
            for (int j = 0; j < 4; j++) minv[icol][j] *= pivinv;
            // Subtract this row from others to zero out their columns
            for (int j = 0; j < 4; j++) {
                if (j != icol) {
                    float save = minv[j][icol];
                    minv[j][icol] = 0;
                    for (int k = 0; k < 4; k++) minv[j][k] -= minv[icol][k]*save;
                }
            }
        }
        // Swap columns to reflect permutation
        for (int j = 3; j >= 0; j--) {
            if (indxr[j] != indxc[j]) {
                for (int k = 0; k < 4; k++) {
                    float temp = minv[k][indxr[j]];
                    minv[k][indxr[j]] = minv[k][indxc[j]];
                    minv[k][indxc[j]] = temp;
                }
            }
        }
        return Matrix4f(minv);
    }

    float m[4][4];
}