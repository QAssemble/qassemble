#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <cppdlr/dlr_build.hpp>
#include <cppdlr/dlr_imfreq.hpp>
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <cppdlr/utils.hpp>
#include <nda/lapack.hpp>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>

namespace nb = nanobind;
using namespace nb::literals;
using RF = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Values = nb::ndarray<const cppdlr::dcomplex, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Indices = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using RealValues = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// Own a separate contiguous buffer before any local cppdlr operator is destroyed.
template <typename T, typename A> auto copy_array(A const &a) {
    auto data = std::make_unique<T[]>(a.size());
    if constexpr (A::rank == 1) {
        for (long i = 0; i < a.size(); ++i) data[i] = a(i);
    } else {
        for (long i = 0; i < a.extent(0); ++i)
            for (long j = 0; j < a.extent(1); ++j) data[i * a.extent(1) + j] = a(i, j);
    }
    T *ptr = data.get();
    nb::capsule owner(ptr, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    data.release();
    if constexpr (A::rank == 1)
        return nb::ndarray<nb::numpy, T, nb::ndim<1>>(ptr, {size_t(a.size())}, owner);
    else
        return nb::ndarray<nb::numpy, T, nb::ndim<2>>(
            ptr, {size_t(a.extent(0)), size_t(a.extent(1))}, owner);
}

void check_lambda(double lamb) {
    if (!std::isfinite(lamb) || lamb <= 0 || lamb > double(INT_MAX / 2 - 1))
        throw std::invalid_argument("lamb must be finite, positive, and fit cppdlr's integer grid");
}

void check_eps(double eps) {
    if (!std::isfinite(eps) || eps <= 0 || eps >= 1)
        throw std::invalid_argument("eps must be finite and between zero and one");
}

void check_paired_rf(RF rf) {
    if (rf.size() == 0) throw std::invalid_argument("rf must not be empty");
    for (size_t j = 0; j < rf.size(); ++j)
        if (!std::isfinite(rf(j)) || rf(j) < 0 || (j && rf(j) <= rf(j - 1)))
            throw std::invalid_argument("paired rf must be finite, nonnegative and strictly increasing");
}

double paired_tau(double t, double w) {
    return w == 0 ? cppdlr::k_it(t, 0) : cppdlr::k_it(t, w) + cppdlr::k_it(t, -w);
}

double paired_freq(int n, double w) {
    if (w == 0) return std::real(cppdlr::k_if(n, 0, cppdlr::Boson));
    return std::real(cppdlr::k_if(n, w, cppdlr::Boson) + cppdlr::k_if(-n, w, cppdlr::Boson));
}

struct RealLU {
    nda::matrix<double, nda::F_layout> lu;
    nda::vector<int> piv;

    explicit RealLU(RealValues a) : lu(a.shape(0), a.shape(1)), piv(a.shape(0)) {
        if (!a.shape(0) || a.shape(0) != a.shape(1))
            throw std::invalid_argument("RealLU requires a nonempty square matrix");
        for (size_t i = 0; i < a.shape(0); ++i)
            for (size_t j = 0; j < a.shape(1); ++j) {
                if (!std::isfinite(a(i, j))) throw std::invalid_argument("matrix must be finite");
                lu(i, j) = a(i, j);
            }
        int info = nda::lapack::getrf(lu, piv);
        if (info != 0) throw std::runtime_error("getrf failed, info = " + std::to_string(info));
    }

    auto solve(RealValues b) const {
        if (b.shape(0) != size_t(lu.extent(0)))
            throw std::invalid_argument("rhs rows must match LU size");
        auto x = nda::matrix<double, nda::F_layout>(b.shape(0), b.shape(1));
        for (size_t i = 0; i < b.shape(0); ++i)
            for (size_t j = 0; j < b.shape(1); ++j) {
                if (!std::isfinite(b(i, j))) throw std::invalid_argument("rhs must be finite");
                x(i, j) = b(i, j);
            }
        int info = nda::lapack::getrs(lu, x, piv);
        if (info != 0) throw std::runtime_error("getrs failed, info = " + std::to_string(info));
        return copy_array<double>(x);
    }
};

nda::vector<double> copy_rf(double lamb, RF rf, bool symmetrize) {
    check_lambda(lamb);
    if (rf.size() == 0) throw std::invalid_argument("rf must not be empty");
    auto out = nda::vector<double>(rf.size());
    for (size_t i = 0; i < rf.size(); ++i) {
        out(i) = rf(i);
        if (!std::isfinite(out(i)) || std::abs(out(i)) > lamb || (i && out(i) <= out(i - 1)))
            throw std::invalid_argument("rf must be finite, strictly increasing, and within [-lamb, lamb]");
    }
    if (symmetrize) {
        if (rf.size() % 2) throw std::invalid_argument("symmetric rf must have even length");
        for (size_t i = 0; i < rf.size(); ++i)
            if (out(i) != -out(rf.size() - 1 - i))
                throw std::invalid_argument("symmetric rf must contain exact +/- pairs");
    }
    return out;
}

nda::matrix<cppdlr::dcomplex> copy_values(Values values, long rows) {
    if (values.shape(0) != size_t(rows))
        throw std::invalid_argument("first axis does not match the operator's nodes or rank");
    auto out = nda::matrix<cppdlr::dcomplex>(rows, values.shape(1));
    for (long i = 0; i < rows; ++i)
        for (size_t j = 0; j < values.shape(1); ++j) {
            auto value = values(i, j);
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                throw std::invalid_argument("values must be finite");
            out(i, j) = value;
        }
    return out;
}

void check_beta(double beta) {
    if (!std::isfinite(beta) || beta <= 0)
        throw std::invalid_argument("beta must be finite and positive");
}

NB_MODULE(_core, m) {
    m.def("build_dlr_rf", [](double lamb, double eps, bool symmetrize) {
        check_lambda(lamb);
        check_eps(eps);
        return copy_array<double>(cppdlr::build_dlr_rf(lamb, eps, symmetrize));
    }, "lamb"_a, "eps"_a, "symmetrize"_a = false);

    nb::class_<RealLU>(m, "RealLU")
        .def(nb::init<RealValues>(), "a"_a)
        .def("solve", &RealLU::solve, "b"_a);

    m.def("paired_boson_nodes", [](double lamb, double eps) {
        check_lambda(lamb);
        check_eps(eps);
        auto fine = cppdlr::fineparams(lamb);
        auto [t, weights] = cppdlr::build_it_fine(fine);
        auto poles = cppdlr::build_rf_fine(fine);
        long half = fine.nom / 2;
        auto candidates = nda::vector<double>(half + 1);
        candidates(0) = 0;
        for (long j = 0; j < half; ++j) candidates(j + 1) = poles(half + j);
        auto weighted = nda::matrix<double>(candidates.size(), fine.nt);
        for (long j = 0; j < candidates.size(); ++j)
            for (long i = 0; i < fine.nt; ++i)
                weighted(j, i) = weights(i) * paired_tau(t(i), candidates(j));
        auto [q1, norms1, piv1] = cppdlr::pivrgs(weighted, eps);
        long rank = norms1.size();
        if (rank == 0) throw std::runtime_error("paired kernel has zero numerical rank");
        std::sort(piv1.begin(), piv1.end());
        auto rf = nda::vector<double>(rank);
        for (long j = 0; j < rank; ++j) rf(j) = candidates(piv1(j));

        auto kt = nda::matrix<double>(fine.nt / 2, rank);
        for (long i = 0; i < fine.nt / 2; ++i)
            for (long j = 0; j < rank; ++j) kt(i, j) = paired_tau(t(i), rf(j));
        auto [q2, norms2, piv2] = cppdlr::pivrgs(kt, 1e-100);
        if (norms2.size() != rank) throw std::runtime_error("paired time kernel lost rank");
        std::sort(piv2.begin(), piv2.end());
        auto t_half = nda::vector<double>(rank);
        for (long j = 0; j < rank; ++j) t_half(j) = t(piv2(j));

        auto kf = nda::matrix<double>(fine.nmax + 1, rank);
        for (int n = 0; n <= fine.nmax; ++n)
            for (long j = 0; j < rank; ++j) kf(n, j) = paired_freq(n, rf(j));
        auto [q3, norms3, piv3] = cppdlr::pivrgs(kf, 1e-100);
        if (norms3.size() != rank) throw std::runtime_error("paired frequency kernel lost rank");
        std::sort(piv3.begin(), piv3.end());
        auto n_half = nda::vector<std::int64_t>(rank);
        for (long j = 0; j < rank; ++j) n_half(j) = piv3(j);
        return std::make_tuple(copy_array<double>(rf), copy_array<double>(t_half),
                               copy_array<std::int64_t>(n_half));
    }, "lamb"_a, "eps"_a);

    m.def("paired_boson_k_tau", [](RF t_rel, RF rf) {
        check_paired_rf(rf);
        auto out = nda::matrix<double>(t_rel.size(), rf.size());
        for (size_t i = 0; i < t_rel.size(); ++i) {
            double t = t_rel(i);
            if (!std::isfinite(t) || !((t >= -0.5 && t <= 0.5) || t == 1))
                throw std::invalid_argument("relative times must lie in [-0.5, 0.5] or equal endpoint +1");
            for (size_t j = 0; j < rf.size(); ++j) out(i, j) = paired_tau(t, rf(j));
        }
        return copy_array<double>(out);
    }, "t_rel"_a, "rf"_a);

    m.def("paired_boson_k_freq", [](Indices n, RF rf) {
        check_paired_rf(rf);
        auto out = nda::matrix<double>(n.size(), rf.size());
        for (size_t i = 0; i < n.size(); ++i) {
            if (n(i) < -(INT_MAX - 1) / 2 || n(i) > (INT_MAX - 1) / 2)
                throw std::invalid_argument("Matsubara index exceeds paired kernel's safe integer range");
            for (size_t j = 0; j < rf.size(); ++j) out(i, j) = paired_freq(int(n(i)), rf(j));
        }
        return copy_array<double>(out);
    }, "n"_a.noconvert(), "rf"_a);
    nb::class_<cppdlr::imtime_ops>(m, "ImTime")
        .def("__init__", [](cppdlr::imtime_ops *self, double lamb, RF rf, bool symmetrize) {
            auto poles = copy_rf(lamb, rf, symmetrize);
            new (self) cppdlr::imtime_ops(lamb, poles, symmetrize);
        }, "lamb"_a, "rf"_a, "symmetrize"_a = false)
        .def_prop_ro("rank", &cppdlr::imtime_ops::rank)
        .def_prop_ro("nodes", [](cppdlr::imtime_ops const &self) {
            return copy_array<double>(self.get_itnodes());
        }, nb::rv_policy::reference)
        .def("vals2coefs", [](cppdlr::imtime_ops const &self, Values g) {
            return copy_array<cppdlr::dcomplex>(self.vals2coefs(copy_values(g, self.rank())));
        }, "g"_a)
        .def("coefs2vals", [](cppdlr::imtime_ops const &self, Values c) {
            return copy_array<cppdlr::dcomplex>(self.coefs2vals(copy_values(c, self.rank())));
        }, "c"_a)
        .def("coefs2eval", [](cppdlr::imtime_ops const &self, Values c, RF t_rel) {
            auto coefficients = copy_values(c, self.rank());
            auto out = nda::matrix<cppdlr::dcomplex>(t_rel.size(), c.shape(1));
            for (size_t i = 0; i < t_rel.size(); ++i) {
                double t = t_rel(i);
                if (!std::isfinite(t) || !((t >= -0.5 && t <= 0.5) || t == 1.0))
                    throw std::invalid_argument("relative times must lie in [-0.5, 0.5] or equal endpoint +1");
                auto row = nda::make_regular(self.coefs2eval(coefficients, t));
                for (size_t j = 0; j < c.shape(1); ++j) out(i, j) = row(j);
            }
            return copy_array<cppdlr::dcomplex>(out);
        }, "c"_a, "t_rel"_a);

    nb::class_<cppdlr::imfreq_ops>(m, "ImFreq")
        .def("__init__", [](cppdlr::imfreq_ops *self, double lamb, RF rf, bool boson, bool symmetrize) {
            auto poles = copy_rf(lamb, rf, symmetrize);
            new (self) cppdlr::imfreq_ops(lamb, poles, boson ? cppdlr::Boson : cppdlr::Fermion, symmetrize);
        }, "lamb"_a, "rf"_a, "boson"_a, "symmetrize"_a = false)
        .def_prop_ro("rank", &cppdlr::imfreq_ops::rank)
        .def_prop_ro("nodes", [](cppdlr::imfreq_ops const &self) {
            return copy_array<int>(self.get_ifnodes());
        }, nb::rv_policy::reference)
        .def("vals2coefs", [](cppdlr::imfreq_ops const &self, double beta, Values g) {
            check_beta(beta);
            return copy_array<cppdlr::dcomplex>(self.vals2coefs(beta, copy_values(g, self.get_ifnodes().size())));
        }, "beta"_a, "g"_a)
        .def("coefs2vals", [](cppdlr::imfreq_ops const &self, double beta, Values c) {
            check_beta(beta);
            return copy_array<cppdlr::dcomplex>(self.coefs2vals(beta, copy_values(c, self.rank())));
        }, "beta"_a, "c"_a)
        .def("coefs2eval", [](cppdlr::imfreq_ops const &self, double beta, Values c, Indices n) {
            check_beta(beta);
            auto coefficients = copy_values(c, self.rank());
            auto out = nda::matrix<cppdlr::dcomplex>(n.size(), c.shape(1));
            for (size_t i = 0; i < n.size(); ++i) {
                if (n(i) < INT_MIN / 2 || n(i) > (INT_MAX - 1) / 2)
                    throw std::invalid_argument("Matsubara index exceeds cppdlr's safe integer range");
                auto row = nda::make_regular(self.coefs2eval(beta, coefficients, static_cast<int>(n(i))));
                for (size_t j = 0; j < c.shape(1); ++j) out(i, j) = row(j);
            }
            return copy_array<cppdlr::dcomplex>(out);
        }, "beta"_a, "c"_a, "n"_a.noconvert());
}
