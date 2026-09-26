#ifndef PLAN_ENV_ROLLING_WINDOW_H
#define PLAN_ENV_ROLLING_WINDOW_H

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

/**
 * A box of voxels of fixed size that follows the drone (SwarmDeck).
 *
 * Voxels have global integer indices, floor(position / resolution), so a
 * voxel keeps its index as the window moves. Each index is stored at its
 * index modulo the window size along each axis: the buffers never move or
 * grow, whatever the size of the mission. Recentring returns the boxes of
 * indices that have just entered; their storage held voxels that have just
 * left, and the caller clears it.
 */
class RollingWindow
{
public:
  /** Inclusive min and max global index. */
  using Box = std::pair<Eigen::Vector3i, Eigen::Vector3i>;

  RollingWindow() = default;
  RollingWindow(double resolution, const Eigen::Vector3i &size)
      : resolution_(resolution), inv_resolution_(1.0 / resolution),
        size_(size.cwiseMax(Eigen::Vector3i::Ones())) {}

  double resolution() const { return resolution_; }
  const Eigen::Vector3i &size() const { return size_; }
  std::size_t cells() const
  {
    return std::size_t(size_(0)) * std::size_t(size_(1)) * std::size_t(size_(2));
  }
  const Eigen::Vector3i &minIndex() const { return min_; }
  Eigen::Vector3i maxIndex() const { return min_ + size_ - Eigen::Vector3i::Ones(); }

  Eigen::Vector3i indexOf(const Eigen::Vector3d &pos) const
  {
    return Eigen::Vector3i(int(std::floor(pos(0) * inv_resolution_)),
                           int(std::floor(pos(1) * inv_resolution_)),
                           int(std::floor(pos(2) * inv_resolution_)));
  }

  Eigen::Vector3d centerOf(const Eigen::Vector3i &index) const
  {
    return (index.cast<double>() + Eigen::Vector3d::Constant(0.5)) * resolution_;
  }

  bool contains(const Eigen::Vector3i &index) const
  {
    return (index.array() >= min_.array()).all() &&
           (index.array() <= maxIndex().array()).all();
  }

  bool contains(const Eigen::Vector3d &pos) const { return contains(indexOf(pos)); }

  Eigen::Vector3i clamp(const Eigen::Vector3i &index) const
  {
    return index.cwiseMax(min_).cwiseMin(maxIndex());
  }

  /** Storage address of an index; only meaningful when contains(index). */
  std::size_t address(const Eigen::Vector3i &index) const
  {
    const std::size_t x = wrap(index(0), size_(0));
    const std::size_t y = wrap(index(1), size_(1));
    const std::size_t z = wrap(index(2), size_(2));
    return (x * std::size_t(size_(1)) + y) * std::size_t(size_(2)) + z;
  }

  Eigen::Vector3d minBound() const { return min_.cast<double>() * resolution_; }
  Eigen::Vector3d maxBound() const
  {
    return (maxIndex() + Eigen::Vector3i::Ones()).cast<double>() * resolution_;
  }

  /** Centre the window on `pos`; returns the boxes of indices that entered. */
  std::vector<Box> recenter(const Eigen::Vector3d &pos)
  {
    const Eigen::Vector3i next = indexOf(pos) - size_ / 2;
    std::vector<Box> entered;
    if (!placed_ || ((next - min_).cwiseAbs().array() >= size_.array()).any())
    {
      placed_ = true;
      min_ = next;
      entered.emplace_back(min_, maxIndex());
      return entered;
    }
    if (next == min_)
      return entered;
    const Eigen::Vector3i old_min = min_;
    const Eigen::Vector3i old_max = maxIndex();
    min_ = next;
    const Eigen::Vector3i new_max = maxIndex();
    for (int axis = 0; axis < 3; ++axis)
    {
      const int shift = next(axis) - old_min(axis);
      if (shift == 0)
        continue;
      Eigen::Vector3i lo = min_;
      Eigen::Vector3i hi = new_max;
      if (shift > 0)
        lo(axis) = old_max(axis) + 1;
      else
        hi(axis) = old_min(axis) - 1;
      entered.emplace_back(lo, hi);
    }
    return entered;
  }

private:
  static std::size_t wrap(int value, int modulus)
  {
    const int remainder = value % modulus;
    return std::size_t(remainder < 0 ? remainder + modulus : remainder);
  }

  double resolution_ = 0.1;
  double inv_resolution_ = 10.0;
  Eigen::Vector3i size_ = Eigen::Vector3i::Ones();
  Eigen::Vector3i min_ = Eigen::Vector3i::Zero();
  bool placed_ = false;
};

#endif
