#pragma once 

#include "Vector.h"
#include "Matrix.h"

struct Vertex : public Vector3f 
{
    Vector3f _normal;

    Vertex(float x, float y, float z, float nx, float ny, float nz) : Vector3f(x, y, z), _normal(Vector3f(nx, ny, nz)) {}
};

struct Triangle {
    unsigned _idx1;
    unsigned _idx2;
    unsigned _idx3;
};