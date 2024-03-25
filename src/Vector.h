#pragma once

#include <assert.h>
#include <cmath>
#include <iostream>

template <typename T> class Vector2 {
public:
    __host__ __device__ Vector2() { x = y = 0; }
    __host__ __device__ Vector2(T x, T y) : x(x), y(y) {}
    __host__ __device__ Vector2(T xx) : x(xx), y(xx) {}
    __host__ __device__ Vector2(const Vector2<T> &v) : x(v.x), y(v.y) {}

    __host__ __device__ bool HasNaNs() const {
        return std::isnan(x) || std::isnan(y);
    }

    __host__ __device__ Vector2<T> &operator=(const Vector2<T> &v) {
        x = v.x;
        y = v.y;
        return *this;
    }

    __host__ __device__ Vector2<T> operator+(const Vector2<T> &v) const {
        return Vector2(x + v.x, y + v.y);
    }

    __host__ __device__ Vector2<T> &operator+=(const Vector2<T> &v) {
        x += v.x;
        y += v.y;
        return *this;
    }

    __host__ __device__ Vector2<T> operator-(const Vector2<T> &v) const {
        return Vector2(x - v.x, y - v.y);
    }

    __host__ __device__ Vector2<T> &operator-=(const Vector2<T> &v) {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    __host__ __device__ Vector2<T> operator*(T s) const {
        return Vector2(x * s, y * s);
    }

    __host__ __device__ Vector2<T> &operator*=(T s) {
        x *= s;
        y *= s;
        return *this;
    }

    __host__ __device__ Vector2<T> operator*(const Vector2<T> &v) const {
        return Vector2(x * v.x, y * v.y);
    }

    __host__ __device__ Vector2<T> &operator*=(const Vector2<T> &v) {
        x *= v.x;
        y *= v.y;
        return *this;
    }

    __host__ __device__ Vector2<T> operator/(T s) const {
        assert(s != 0);
        T inv = 1 / s;
        return Vector2(x * inv, y * inv);
    }

    __host__ __device__ Vector2<T> &operator/=(T s) {
        assert(s != 0);
        T inv = 1 / s;
        x *= inv;
        y *= inv;
        return *this;
    }

    __host__ __device__ Vector2<T> operator-() const {
        return Vector2(-x, -y);
    }

    __host__ __device__ bool operator==(const Vector2<T> &v) const {
        return x == v.x && y == v.y;
    }

    __host__ __device__ bool operator!=(const Vector2<T> &v) const {
        return x != v.x || y != v.y;
    }

    __host__ __device__ T operator[](int i) const {
        assert(i >= 0 && i <= 1);
        return i == 0 ? x : y;
    }

    __host__ __device__ T &operator[](int i) {
        assert(i >= 0 && i <= 1);
        return i == 0 ? x : y;
    }

    inline __host__ __device__ float LengthSquared() const {
        return x * x + y * y;
    }

    inline __host__ __device__ float Length() const {
        return std::sqrt(LengthSquared());
    }

    friend std::ostream &operator<<(std::ostream &os, const Vector2<T> &v) {
        os << "[" << v.x << ", " << v.y << "]";
        return os;
    }

public:
    T x, y;
};

template <typename T> class Vector3 {
public:
    __host__ __device__ Vector3() { x = y = z = 0; }
    __host__ __device__ Vector3(T x, T y, T z) : x(x), y(y), z(z) {}
    __host__ __device__ Vector3(T xx) : x(xx), y(xx), z(xx) {}
    __host__ __device__ Vector3(const Vector3<T> &v) : x(v.x), y(v.y), z(v.z) {}

    __host__ __device__ bool HasNaNs() const {
        return std::isnan(x) || std::isnan(y) || std::isnan(z);
    }

    __host__ __device__ Vector3<T> &operator=(const Vector3<T> &v) {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }

    __host__ __device__ Vector3<T> operator+(const Vector3<T> &v) const {
        return Vector3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Vector3<T> &operator+=(const Vector3<T> &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    __host__ __device__ Vector3<T> operator-(const Vector3<T> &v) const {
        return Vector3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ Vector3<T> &operator-=(const Vector3<T> &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    __host__ __device__ Vector3<T> operator*(T s) const {
        return Vector3(x * s, y * s, z * s);
    }

    __host__ __device__ Vector3<T> &operator*=(T s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    __host__ __device__ Vector3<T> operator*(const Vector3<T> &v) const {
        return Vector3(x * v.x, y * v.y, z * v.z);
    }

    __host__ __device__ Vector3<T> &operator*=(const Vector3<T> &v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    __host__ __device__ Vector3<T> operator/(T s) const {
        assert(s != 0);
        T inv = 1 / s;
        return Vector3(x * inv, y * inv, z * inv);
    }

    __host__ __device__ Vector3<T> &operator/=(T s) {
        assert(s != 0);
        T inv = 1 / s;
        x *= inv;
        y *= inv;
        z *= inv;
        return *this;
    }

    __host__ __device__ Vector3<T> operator-() const {
        return Vector3(-x, -y, -z);
    }

    __host__ __device__ bool operator==(const Vector3<T> &v) const {
        return x == v.x && y == v.y && z == v.z;
    }

    __host__ __device__ bool operator!=(const Vector3<T> &v) const {
        return x != v.x || y != v.y || z != v.z;
    }

    __host__ __device__ T operator[](int i) const {
        assert(i >= 0 && i <= 2);
        return i == 0 ? x : (i == 1 ? y : z);
    }

    __host__ __device__ T &operator[](int i) {
        assert(i >= 0 && i <= 2);
        return i == 0 ? x : (i == 1 ? y : z);
    }

    inline __host__ __device__ float LengthSquared() const {
        return x * x + y * y + z * z;
    }

    inline __host__ __device__ float Length() const {
        return std::sqrt(LengthSquared());
    }

    inline __host__ __device__ T MaxComponent() const {
        return std::max(x, std::max(y, z));
    }

    inline __host__ __device__ T MinComponent() const {
        return std::min(x, std::min(y, z));
    }

    friend std::ostream &operator<<(std::ostream &os, const Vector3<T> &v) {
        os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
        return os;
    }

public:
    T x, y, z;
};

// Vector3 Utility Functions

template <typename T> inline __host__ __device__ Vector3<T> Abs(const Vector3<T> &v) {
    return Vector3<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z));
}

template <typename T> inline __host__ __device__ T Dot(const Vector3<T> &v1, const Vector3<T> &v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template <typename T> inline __host__ __device__ T AbsDot(const Vector3<T> &v1, const Vector3<T> &v2) {
    return std::abs(Dot(v1, v2));
}

template <typename T> inline __host__ __device__ Vector3<T> Cross(const Vector3<T> &v1, const Vector3<T> &v2) {
    return Vector3<T>((v1.y * v2.z) - (v1.z * v2.y),
                      (v1.z * v2.x) - (v1.x * v2.z),
                      (v1.x * v2.y) - (v1.y * v2.x));
}

template <typename T> inline __host__ __device__ Vector3<T> Normalize(const Vector3<T> &v) {
    return v / v.Length();
}

template <typename T> inline __host__ __device__ Vector3<T> Min(const Vector3<T> &v1, const Vector3<T> &v2) {
    return Vector3<T>(std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z));
}

template <typename T> inline __host__ __device__ Vector3<T> Max(const Vector3<T> &v1, const Vector3<T> &v2) {
    return Vector3<T>(std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z));
}

template <typename T> inline __host__ __device__ Vector3<T> Permute(const Vector3<T> &v, int x, int y, int z) {
    return Vector3<T>(v[x], v[y], v[z]);
}

template <typename T> inline __host__ __device__ void CoordinateSystem(const Vector3<T> &v1, Vector3<T> *v2, Vector3<T> *v3) {
    if (std::abs(v1.x) > std::abs(v1.y)) {
        *v2 = Vector3<T>(-v1.z, 0, v1.x) / std::sqrt(v1.x * v1.x + v1.z * v1.z);
    } else {
        *v2 = Vector3<T>(0, v1.z, -v1.y) / std::sqrt(v1.y * v1.y + v1.z * v1.z);
    }
    *v3 = Cross(v1, *v2);
}

template <typename T> inline __host__ __device__ float Distance(const Vector3<T> &p1, const Vector3<T> &p2) {
    return (p1 - p2).Length();
}

template <typename T> inline __host__ __device__ float DistanceSquared(const Vector3<T> &p1, const Vector3<T> &p2) {
    return (p1 - p2).LengthSquared();
}

template <typename T> inline __host__ __device__ Vector3<T> Lerp(float t, const Vector3<T> &v1, const Vector3<T> &v2) {
    return (1 - t) * v1 + t * v2;
}

template <typename T> inline __host__ __device__ Vector3<T> FaceForward(const Vector3<T> &v1, const Vector3<T> &v2) {
    return (Dot(v1, v2) < 0) ? -v1 : v1;
}

typedef Vector2<float> Vector2f;
typedef Vector2<int> Vector2i;
typedef Vector3<float> Vector3f;
typedef Vector3<int> Vector3i;

template <typename T> class Vector4 {
public:
    __host__ __device__ Vector4() { x = y = z = w = 0; }
    __host__ __device__ Vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
    __host__ __device__ Vector4(T xx) : x(xx), y(xx), z(xx), w(xx) {}
    __host__ __device__ Vector4(const Vector4<T> &v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

    __host__ __device__ bool HasNaNs() const {
        return std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isnan(w);
    }

    __host__ __device__ Vector4<T> &operator=(const Vector4<T> &v) {
        x = v.x;
        y = v.y;
        z = v.z;
        w = v.w;
        return *this;
    }

    __host__ __device__ Vector4<T> operator+(const Vector4<T> &v) const {
        return Vector4(x + v.x, y + v.y, z + v.z, w + v.w);
    }

    __host__ __device__ Vector4<T> &operator+=(const Vector4<T> &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }

    __host__ __device__ Vector4<T> operator-(const Vector4<T> &v) const {
        return Vector4(x - v.x, y - v.y, z - v.z, w - v.w);
    }

    __host__ __device__ Vector4<T> &operator-=(const Vector4<T> &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return *this;
    }

    __host__ __device__ Vector4<T> operator*(T s) const {
        return Vector4(x * s, y * s, z * s, w * s);
    }

    __host__ __device__ Vector4<T> &operator*=(T s) {
        x *= s;
        y *= s;
        z *= s;
        w *= s;
        return *this;
    }

    __host__ __device__ Vector4<T> operator*(const Vector4<T> &v) const {
        return Vector4(x * v.x, y * v.y, z * v.z, w * v.w);
    }

    __host__ __device__ Vector4<T> &operator*=(const Vector4<T> &v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        w *= v.w;
        return *this;
    }

    __host__ __device__ Vector4<T> operator/(T s) const {
        assert(s != 0);
        T inv = 1 / s;
        return Vector4(x * inv, y * inv, z * inv, w * inv);
    }

    __host__ __device__ Vector4<T> &operator/=(T s) {
        assert(s != 0);
        T inv = 1 / s;
        x *= inv;
        y *= inv;
        z *= inv;
        w *= inv;
        return *this;
    }

    __host__ __device__ Vector4<T> operator-() const {
        return Vector4(-x, -y, -z, -w);
    }

    __host__ __device__ bool operator==(const Vector4<T> &v) const {
        return x == v.x && y == v.y && z == v.z && w == v.w;
    }

    __host__ __device__ bool operator!=(const Vector4<T> &v) const {
        return x != v.x || y != v.y || z != v.z || w != v.w;
    }

    __host__ __device__ T operator[](int i) const {
        assert(i >= 0 && i <= 3);
        return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
    }

    __host__ __device__ T &operator[](int i) {
        assert(i >= 0 && i <= 3);
        return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
    }

    inline __host__ __device__ float LengthSquared() const {
        return x * x + y * y + z * z + w * w;
    }

    inline __host__ __device__ float Length() const {
        return std::sqrt(LengthSquared());
    }

    inline __host__ __device__ T MaxComponent() const {
        return std::max(x, std::max(y, std::max(z, w)));
    }

    inline __host__ __device__ T MinComponent() const {
        return std::min(x, std::min(y, std::min(z, w)));
    }

    friend std::ostream &operator<<(std::ostream &os, const Vector4<T> &v) {
        os << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
        return os;
    }

public:
    T x, y, z, w;
};

typedef Vector4<float> Vector4f;
typedef Vector4<int> Vector4i;