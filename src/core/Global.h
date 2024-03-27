#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <memory>
#include <cstdlib>
#include <string.h>
#include <chrono>

// Class Declarations

class Transform;
class Ray;
class RGBSpectrum;
typedef RGBSpectrum Spectrum;
class Object;
class Light;
struct VisibilityTester;
class Sampler;
class Camera;
class Material;
class Film;
struct HitRecord;
class Scene;
class Renderer;
class Fresnel;
class BxDF;
class BSDF;
class Medium;
class MicrofacetDistribution;
class PhaseFunction;
struct MediumRecord;
enum class TransportMode;

// Constants

const float INF = std::numeric_limits<float>::infinity();
static constexpr float Infinity = std::numeric_limits<float>::infinity();
static constexpr float MaxFloat = std::numeric_limits<float>::max();
const double PI = 3.1415926535897932385;
const double invPI = 1.f / PI;
const float Inv4Pi = 0.07957747154594766788;
const float PiOver2 = 1.57079632679489661923;
const float PiOver4 = 0.78539816339744830961;
const float ShadowEpsilon = 0.0001f;

// Utility Functions

template <typename T, typename U, typename V>
inline __host__ __device__ T Clamp(T val, U low, V high) {
    if (val < low) return low;
    else if (val > high) return high;
    else return val;
}

inline __host__ __device__ float Lerp(float t, float v1, float v2) {
    return (1 - t) * v1 + t * v2;
}

inline __host__ __device__ float DegreesToRadicn(float deg) {
    return deg * PI / 180.f;
}