#pragma once

#include <algorithm>
#include <cmath>
#include <bspline_opt/uniform_bspline.h>
#include <plan_env/flight_band_check.h>

namespace ego_planner
{
// Caller retains GridMap::lock() through publication. Repeat the existing
// optimizer's prefix sampling and the ceiling lane's full-tail band policy
// against one map generation; do not widen upstream's obstacle policy.
  inline bool trajectoryPublicationSafe(GridMap & map, UniformBspline & trajectory)
  {
    double begin, end;
    trajectory.getTimeSpan(begin, end);
    const double step = (end - begin) /
      ((trajectory.evaluateDeBoorT(end) - trajectory.evaluateDeBoorT(begin)).norm() /
      map.getResolution());
    for (double t = begin; t < end * 2 / 3; t += step) {
      if (map.getInflateOccupancy(trajectory.evaluateDeBoorT(t))) {
        return false;
      }
    }

    double low, high;
    if (map.flightBandLimits(trajectory.evaluateDeBoorT(0), low, high)) {
      FlightBandTrajectoryCheck band_check(map);
      const double duration = trajectory.getTimeSum();
      const int samples = std::max(1, int(std::ceil(duration / 0.01)));
      for (int i = 0; i <= samples; ++i) {
        const double time = duration * i / samples;
        if (!band_check.accept(trajectory.evaluateDeBoorT(time), time, i == samples)) {
          return false;
        }
      }
    }
    return true;
  }
} // namespace ego_planner
