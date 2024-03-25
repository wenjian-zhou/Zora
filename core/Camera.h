#pragma once 

#include "Geometry.h"
#include "Vector.h"
#include "Matrix.h"

#define M_PI 3.14156265
#define PI_OVER_TWO 1.5707963267948966192313216916397514420985

// Camera struct, used to store interactive camera data, copied to the GPU and used by CUDA for each frame
struct Camera {
	Vector2f resolution;
	Vector3f position;
	Vector3f view;
	Vector3f up;
	Vector2f fov;
	float apertureRadius;
	float focalDistance;
};

// class for interactive camera object, updated on the CPU for each frame and copied into Camera struct
class InteractiveCamera
{
private:

	Vector3f centerPosition;
	Vector3f viewDirection;
	float yaw;
	float pitch;
	float radius;
	float apertureRadius;
	float focalDistance;

	void FixYaw();
	void FixPitch();
	void FixRadius();
	void FixApertureRadius();
	void FixFocalDistance();

public:
	InteractiveCamera();
	virtual ~InteractiveCamera();
	void ChangeYaw(float m);
	void ChangePitch(float m);
	void ChangeRadius(float m);
	void ChangeAltitude(float m);
	void ChangeFocalDistance(float m);
	void Strafe(float m);
	void GoForward(float m);
	void RotateRight(float m);
	void ChangeApertureDiameter(float m);
	void SetResolution(float x, float y);
	void SetFOVX(float fovx);

	void BuildRenderCamera(Camera* renderCamera);

	Vector2f resolution;
	Vector2f fov;
};