// The rolling window's index arithmetic (SwarmDeck drone scout design §3.3).

#include <gtest/gtest.h>

#include <set>
#include <tuple>

#include "plan_env/rolling_window.h"

namespace {

std::set<std::size_t> addresses(const RollingWindow& w, int x_from, int x_to) {
  std::set<std::size_t> out;
  for (int x = x_from; x <= x_to; ++x)
    for (int y = w.minIndex()(1); y <= w.maxIndex()(1); ++y)
      for (int z = w.minIndex()(2); z <= w.maxIndex()(2); ++z)
        out.insert(w.address(Eigen::Vector3i(x, y, z)));
  return out;
}

TEST(RollingWindow, EveryVoxelInsideHasItsOwnAddress) {
  RollingWindow w(0.1, Eigen::Vector3i(4, 5, 3));
  w.recenter(Eigen::Vector3d(-1.23, 7.8, 0.4));
  const std::set<std::size_t> all = addresses(w, w.minIndex()(0), w.maxIndex()(0));
  EXPECT_EQ(all.size(), w.cells());
  EXPECT_LT(*all.rbegin(), w.cells());
}

TEST(RollingWindow, IndicesFloorNegativeCoordinates) {
  RollingWindow w(0.1, Eigen::Vector3i(4, 4, 4));
  EXPECT_EQ(w.indexOf(Eigen::Vector3d(-0.05, 0.05, -0.15)), Eigen::Vector3i(-1, 0, -2));
}

TEST(RollingWindow, MovingEntersOnlyNewSlabsInTheStorageOfTheLeavingOnes) {
  RollingWindow w(1.0, Eigen::Vector3i(4, 4, 4));
  w.recenter(Eigen::Vector3d(0.5, 0.5, 0.5));  // indices [-2, 1] on each axis
  const std::set<std::size_t> leaving = addresses(w, -2, -1);
  const auto entered = w.recenter(Eigen::Vector3d(2.5, 0.5, 0.5));  // x in [0, 3]
  ASSERT_EQ(entered.size(), 1u);
  EXPECT_EQ(entered[0].first, Eigen::Vector3i(2, -2, -2));
  EXPECT_EQ(entered[0].second, Eigen::Vector3i(3, 1, 1));
  EXPECT_EQ(addresses(w, 2, 3), leaving);
  EXPECT_FALSE(w.contains(Eigen::Vector3i(-1, 0, 0)));
  EXPECT_TRUE(w.contains(Eigen::Vector3i(3, 1, 1)));
}

TEST(RollingWindow, StayingPutEntersNothingAndAJumpEntersEverything) {
  RollingWindow w(1.0, Eigen::Vector3i(4, 4, 4));
  w.recenter(Eigen::Vector3d(0.5, 0.5, 0.5));
  EXPECT_TRUE(w.recenter(Eigen::Vector3d(0.9, 0.1, 0.6)).empty());
  const auto entered = w.recenter(Eigen::Vector3d(100.5, 0.5, 0.5));
  ASSERT_EQ(entered.size(), 1u);
  EXPECT_EQ(entered[0].first, w.minIndex());
  EXPECT_EQ(entered[0].second, w.maxIndex());
}

TEST(RollingWindow, BoundsAndClampAreTheWindow) {
  RollingWindow w(0.1, Eigen::Vector3i(240, 240, 80));
  w.recenter(Eigen::Vector3d::Zero());
  EXPECT_TRUE(w.minBound().isApprox(Eigen::Vector3d(-12.0, -12.0, -4.0)));
  EXPECT_TRUE(w.maxBound().isApprox(Eigen::Vector3d(12.0, 12.0, 4.0)));
  EXPECT_EQ(w.clamp(Eigen::Vector3i(500, -500, 0)), Eigen::Vector3i(119, -120, 0));
  EXPECT_EQ(w.cells(), 240u * 240u * 80u);
}

TEST(RollingWindow, NegativeShiftReusesTheLeavingStorage) {
  RollingWindow w(1.0, Eigen::Vector3i(4, 4, 4));
  w.recenter(Eigen::Vector3d(0.5, 0.5, 0.5));
  const auto leaving = addresses(w, 0, 1);
  const auto entered = w.recenter(Eigen::Vector3d(-1.5, 0.5, 0.5));
  ASSERT_EQ(entered.size(), 1u);
  EXPECT_EQ(entered[0].first, Eigen::Vector3i(-4, -2, -2));
  EXPECT_EQ(entered[0].second, Eigen::Vector3i(-3, 1, 1));
  EXPECT_EQ(addresses(w, -4, -3), leaving);
}

TEST(RollingWindow, DiagonalShiftEntersExactlyTheDifferenceOfTheWindows) {
  RollingWindow w(1.0, Eigen::Vector3i(4, 5, 3));
  w.recenter(Eigen::Vector3d(0.5, 0.5, 0.5));
  const RollingWindow old = w;
  const auto entered = w.recenter(Eigen::Vector3d(-0.5, 2.5, -0.5));
  ASSERT_EQ(entered.size(), 3u);
  using Index = std::tuple<int, int, int>;
  std::set<Index> actual, expected;
  for (const auto &box : entered)
    for (int x = box.first(0); x <= box.second(0); ++x)
      for (int y = box.first(1); y <= box.second(1); ++y)
        for (int z = box.first(2); z <= box.second(2); ++z)
          actual.emplace(x, y, z);
  for (int x = w.minIndex()(0); x <= w.maxIndex()(0); ++x)
    for (int y = w.minIndex()(1); y <= w.maxIndex()(1); ++y)
      for (int z = w.minIndex()(2); z <= w.maxIndex()(2); ++z)
        if (!old.contains(Eigen::Vector3i(x, y, z)))
          expected.emplace(x, y, z);
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(addresses(w, w.minIndex()(0), w.maxIndex()(0)).size(), w.cells());
}

}  // namespace
