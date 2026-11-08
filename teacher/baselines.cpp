#include "teacher.hpp"

#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/selection/selector.hpp"

#include <algorithm>
#include <format>

namespace teacher {

std::vector<std::size_t> weighted_budgets(const core::TrajectoryDb &db,
                                          const std::vector<double> &w,
                                          std::size_t interior_total) {
    const std::size_t n = db.size();
    std::vector<std::size_t> cap(n), q(n, 0);
    for (std::size_t i = 0; i < n; ++i)
        cap[i] = db.trajectory(i).size() >= 2 ? db.trajectory(i).size() - 2 : 0;
    std::vector<char> open(n, 1);
    std::size_t left = interior_total;
    while (left > 0) {
        double sw = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (open[i])
                sw += w[i];
        if (sw <= 0)
            break;
        std::vector<double> frac(n, 0);
        std::size_t given = 0;
        bool capped = false;
        for (std::size_t i = 0; i < n; ++i) {
            if (!open[i])
                continue;
            const double want = static_cast<double>(left) * w[i] / sw;
            auto add = static_cast<std::size_t>(want);
            if (q[i] + add >= cap[i]) {
                add = cap[i] - q[i];
                open[i] = 0;
                capped = true;
            } else {
                frac[i] = want - static_cast<double>(add);
            }
            q[i] += add;
            given += add;
        }
        left -= given;
        if (capped)
            continue;
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < n; ++i)
            if (open[i])
                idx.push_back(i);
        std::ranges::sort(idx, [&](auto a, auto b) {
            return frac[a] > frac[b] || (frac[a] == frac[b] && a < b);
        });
        for (std::size_t k = 0; k < idx.size() && left > 0; ++k) {
            ++q[idx[k]];
            --left;
            if (q[idx[k]] == cap[idx[k]])
                open[idx[k]] = 0;
        }
        if (given == 0 && idx.empty())
            break;
    }
    for (std::size_t i = 0; i < n; ++i)
        q[i] += std::min<std::size_t>(2, db.trajectory(i).size());
    return q;
}

namespace {

KeptSets select(const core::TrajectoryDb &db, selection::AlgorithmType algorithm,
                const std::vector<std::size_t> &budgets) {
    KeptSets kept(db.size());
    core::parallel_for(db.size(), [&](std::size_t i) {
        kept[i] = selection::simplify_trajectory(db.trajectory(i), algorithm, budgets[i]);
    });
    return kept;
}

} // namespace

std::map<std::string, KeptSets> baseline_arms(const core::TrajectoryDb &db,
                                              const TeacherCaches &caches,
                                              std::span<const double> shares) {
    using selection::AlgorithmType;
    std::vector<char> partnered(db.size(), 0);
    for (const auto &p : caches.pairs)
        if (p.seen)
            partnered[p.i] = partnered[p.j] = 1;

    std::map<std::string, KeptSets> arms;
    for (const double share : shares) {
        const auto B = total_budget(db, share);
        const auto tag = share_tag(share);
        auto geometric = [&](double alpha) {
            return allocation::create_trajectory_budgets(
                db, allocation::AllocatorType::SizeProportional, B, alpha);
        };
        const auto equal = geometric(0.0);
        arms["RDP_a0_" + tag] = select(db, AlgorithmType::RDP, equal);
        arms["RDP_a025_" + tag] = select(db, AlgorithmType::RDP, geometric(0.25));
        arms["SED_a0_" + tag] = select(db, AlgorithmType::SED, equal);
        arms["UNI_a0_" + tag] = select(db, AlgorithmType::UNIFORM, equal);
        for (const double lambda : {0.5, 1.0}) {
            std::vector<double> w(db.size());
            for (std::size_t i = 0; i < db.size(); ++i)
                w[i] = 1.0 + (partnered[i] ? lambda : 0.0);
            arms[std::format("PARTNER_l{:g}_{}", lambda, tag)] =
                select(db, AlgorithmType::RDP, weighted_budgets(db, w, B - 2 * db.size()));
        }
    }
    return arms;
}

} // namespace teacher
