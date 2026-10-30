#pragma once

#include "backend/engine/core/local_point.hpp"
#include <numbers>

namespace core {

// This is hardcoded for the geolife dataset
inline constexpr double origin_lat = 39.95;
inline constexpr double origin_lon = 116.35;

namespace detail {

struct ProjectionCoeffs {
    double lat1;          // m / deg
    double lon1;          // m / deg
    double lon_lat_cross; // m / deg^2 (meridian convergence)
    double lon2;          // m / deg^2 (parallel curvature)
    double lat2;          // m / deg^2 (meridional radius shift)
};

consteval ProjectionCoeffs make_wgs84_coeffs(double lat_deg) noexcept {
    constexpr double deg_to_rad = std::numbers::pi / 180.0;
    constexpr double semi_major_axis = 6378137.0;
    constexpr double eccentricity_sq = 0.00669437999014132;

    const double rad = lat_deg * deg_to_rad;
    double sin_lat = 0.0, cos_lat = 0.0, term_s = rad, term_c = 1.0;
    for (int i = 1; i <= 10; ++i) {
        sin_lat += term_s;
        cos_lat += term_c;
        term_s *= -rad * rad / ((2 * i) * (2 * i + 1));
        term_c *= -rad * rad / ((2 * i - 1) * (2 * i));
    }

    const double w2 = 1.0 - eccentricity_sq * sin_lat * sin_lat;
    double w = w2;
    for (int i = 0; i < 6; ++i) {
        w = 0.5 * (w + w2 / w);
    }

    const double M = semi_major_axis * (1.0 - eccentricity_sq) / (w2 * w);
    const double N = semi_major_axis / w;
    const double deg_to_rad_sq = deg_to_rad * deg_to_rad;

    return {
        .lat1 = M * deg_to_rad,
        .lon1 = N * cos_lat * deg_to_rad,
        .lon_lat_cross = -M * sin_lat * deg_to_rad_sq,
        .lon2 = 0.5 * N * sin_lat * cos_lat * deg_to_rad_sq,
        .lat2 = 1.5 * M * eccentricity_sq * sin_lat * cos_lat / w2 * deg_to_rad_sq,
    };
}

inline constexpr ProjectionCoeffs coeffs = make_wgs84_coeffs(origin_lat);

} // namespace detail

constexpr core::LocalPoint to_local_metres(double lat, double lon) noexcept {
    const double dlat = lat - origin_lat;
    const double dlon = lon - origin_lon;

    return {
        .x = dlon * (detail::coeffs.lon1 + detail::coeffs.lon_lat_cross * dlat),
        .y = dlat * (detail::coeffs.lat1 + detail::coeffs.lat2 * dlat) +
             detail::coeffs.lon2 * (dlon * dlon),
    };
}

struct LatLon {
    double lat;
    double lon;
};

constexpr LatLon to_lat_lon(double x, double y) noexcept {
    double dlat = y / detail::coeffs.lat1;
    double dlon = x / detail::coeffs.lon1;
    for (int i = 0; i < 5; ++i) {
        dlon = x / (detail::coeffs.lon1 + detail::coeffs.lon_lat_cross * dlat);
        dlat = (y - detail::coeffs.lon2 * (dlon * dlon)) /
               (detail::coeffs.lat1 + detail::coeffs.lat2 * dlat);
    }
    return {.lat = origin_lat + dlat, .lon = origin_lon + dlon};
}

} // namespace core