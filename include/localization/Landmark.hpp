#ifndef LANDMARK_H
#define LANDMARK_H

#include <cstdint>
#include <cmath>

class Landmark
{
public:
  // Constructor para landmarks del mapa (posición global conocida)
  Landmark(int i, double X, double Y, double Z)
      : id_(i), x_(X), y_(Y), z_(Z), range_(NAN), bearing_(NAN) {}

  // Constructor para mediciones (range, bearing) sin posición global
  Landmark(int i, double range, double bearing)
      : id_(i), x_(NAN), y_(NAN), z_(NAN), range_(range), bearing_(bearing) {}

  // Getters
  uint16_t id(void) const { return id_; }
  double x(void) const { return x_; }     // sólo válido si es landmark del mapa
  double y(void) const { return y_; }
  double z(void) const { return z_; }

    double range()   const { return range_; }  // sólo válido si es landmark medido
    double bearing() const { return bearing_; }
private:
  double x_;
  double y_;
  double z_;
  uint16_t id_;
  double range_;
  double bearing_; // medición (si viene del sensor)
};

#endif