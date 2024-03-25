#include "../core/Vector.h"

int main() {
    Vector3f v1(1, 2, 3);

    std::cout << "v1: " << Normalize(v1) << std::endl;

    return 0;
}