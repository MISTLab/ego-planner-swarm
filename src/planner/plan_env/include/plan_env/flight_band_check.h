#ifndef PLAN_ENV_FLIGHT_BAND_CHECK_H
#define PLAN_ENV_FLIGHT_BAND_CHECK_H

#include <algorithm>
#include "plan_env/grid_map.h"

// Ordered trajectory samples, including t=0 and the endpoint. A drifted
// start may recover monotonically through the 0.1 m band fringe for at most
// 0.5 s. First entry closes that exception; real obstacles never get it.
class FlightBandTrajectoryCheck
{
public:
  explicit FlightBandTrajectoryCheck(GridMap &map) : map_(map) {}

  bool accept(const Eigen::Vector3d &point, double time, bool endpoint)
  {
    double low, high;
    if (!map_.flightBandLimits(point, low, high)) return true;
    const double violation = std::max({low - point.z(), point.z() - high, 0.0});
    if (violation == 0.0)
    {
      recovering_ = false;
      return true;
    }
    // Only the recovery fringe gets an extra occupancy check. In-band
    // collision checks retain EGO's upstream prefix/replanning policy.
    if (map_.getInflateOccupancy(point, 0.1) != 0) return false;
    if (!recovering_ || endpoint || time > 0.5 ||
        violation > previous_violation_ + 1e-9)
      return false;
    previous_violation_ = violation;
    return true;
  }

private:
  GridMap &map_;
  bool recovering_ = true;
  double previous_violation_ = 0.1;
};

#endif
