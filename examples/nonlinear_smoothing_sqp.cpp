#include "elqcp.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct ExperimentConfig {
    int N = 250;
    double T = 20.0;
    double lambda = 7e-1;
    double noise_sigma = 0.08;
    double y0 = 1.0;

    int max_sqp_iters = 30;
    double grad_tol = 1e-6;
    double rel_cost_tol = 1e-10;

    double ls_beta = 0.5;
    double ls_min_alpha = 1e-5;

    double fd_eps = 1e-6;
    unsigned int seed = 42;

    double dt() const { return T / static_cast<double>(N); }
};

struct FdCheckResult {
    double fd_dir = 0.0;
    double adj_dir = 0.0;
    double rel_error = 0.0;
};

static double rhs_true(double t, double y)
{
    return -0.05 * y * y * y + 1.2 * std::sin(t);
}

static double rhs_model(double /*t*/, double y, double u)
{
    return -0.05 * y * y * y + u;
}

static double drhs_dy(double /*t*/, double y)
{
    return -0.15 * y * y;
}

static std::vector<double> make_time_grid(const ExperimentConfig& cfg)
{
    std::vector<double> t(cfg.N + 1, 0.0);
    const double dt = cfg.dt();
    for (int k = 0; k <= cfg.N; ++k)
        t[k] = dt * static_cast<double>(k);
    return t;
}

static std::vector<double> rollout_state(const ExperimentConfig& cfg,
                                         const std::vector<double>& u)
{
    if (static_cast<int>(u.size()) != cfg.N)
        throw std::runtime_error("rollout_state: invalid control vector size");

    std::vector<double> y(cfg.N + 1, 0.0);
    y[0] = cfg.y0;

    const double dt = cfg.dt();
    for (int k = 0; k < cfg.N; ++k) {
        const double tk = dt * static_cast<double>(k);
        y[k + 1] = y[k] + dt * rhs_model(tk, y[k], u[k]);
    }
    return y;
}

static double compute_objective(const ExperimentConfig& cfg,
                                const std::vector<double>& y,
                                const std::vector<double>& u,
                                const std::vector<double>& y_obs)
{
    if (static_cast<int>(y.size()) != cfg.N + 1 ||
        static_cast<int>(u.size()) != cfg.N ||
        static_cast<int>(y_obs.size()) != cfg.N + 1) {
        throw std::runtime_error("compute_objective: invalid input size");
    }

    double tracking = 0.0;
    for (int k = 0; k <= cfg.N; ++k) {
        const double e = y[k] - y_obs[k];
        tracking += e * e;
    }

    double control_reg = 0.0;
    for (int k = 0; k < cfg.N; ++k)
        control_reg += u[k] * u[k];

    return tracking + cfg.lambda * cfg.dt() * control_reg;
}

static void build_data(const ExperimentConfig& cfg,
                       std::vector<double>& y_true,
                       std::vector<double>& y_obs)
{
    y_true.assign(cfg.N + 1, 0.0);
    y_true[0] = cfg.y0;

    const double dt = cfg.dt();
    for (int k = 0; k < cfg.N; ++k) {
        const double tk = dt * static_cast<double>(k);
        y_true[k + 1] = y_true[k] + dt * rhs_true(tk, y_true[k]);
    }

    y_obs = y_true;
    std::mt19937 gen(cfg.seed);
    std::normal_distribution<double> noise(0.0, cfg.noise_sigma);

    y_obs[0] = y_true[0];
    for (int k = 1; k <= cfg.N; ++k)
        y_obs[k] += noise(gen);
}

static Eigen::VectorXd discrete_adjoint_gradient(const ExperimentConfig& cfg,
                                                 const std::vector<double>& y,
                                                 const std::vector<double>& u,
                                                 const std::vector<double>& y_obs)
{
    Eigen::VectorXd grad(cfg.N);
    std::vector<double> psi(cfg.N + 1, 0.0);

    const double dt = cfg.dt();
    psi[cfg.N] = 2.0 * (y[cfg.N] - y_obs[cfg.N]);

    for (int k = cfg.N - 1; k >= 0; --k) {
        grad(k) = 2.0 * cfg.lambda * dt * u[k] + dt * psi[k + 1];

        const double tk = dt * static_cast<double>(k);
        const double Ak = 1.0 + dt * drhs_dy(tk, y[k]);
        psi[k] = 2.0 * (y[k] - y_obs[k]) + Ak * psi[k + 1];
    }

    return grad;
}

static ELQCP build_increment_qp(const ExperimentConfig& cfg,
                                const std::vector<double>& y,
                                const std::vector<double>& u,
                                const std::vector<double>& y_obs)
{
    ELQCP prob;
    prob.N = cfg.N;
    prob.nx = 1;
    prob.nu = 1;

    prob.Q.assign(cfg.N, Eigen::MatrixXd::Zero(1, 1));
    prob.M.assign(cfg.N, Eigen::MatrixXd::Zero(1, 1));
    prob.R.assign(cfg.N, Eigen::MatrixXd::Zero(1, 1));
    prob.q.assign(cfg.N, Eigen::VectorXd::Zero(1));
    prob.r.assign(cfg.N, Eigen::VectorXd::Zero(1));
    prob.f.assign(cfg.N, 0.0);
    prob.A.assign(cfg.N, Eigen::MatrixXd::Zero(1, 1));
    prob.B.assign(cfg.N, Eigen::MatrixXd::Zero(1, 1));
    prob.b.assign(cfg.N, Eigen::VectorXd::Zero(1));

    const double dt = cfg.dt();
    for (int k = 0; k < cfg.N; ++k) {
        const double tk = dt * static_cast<double>(k);
        const double fy = drhs_dy(tk, y[k]);

        prob.A[k](0, 0) = 1.0 + dt * fy;
        prob.B[k](0, 0) = dt;

        const double defect = y[k] + dt * rhs_model(tk, y[k], u[k]) - y[k + 1];
        prob.b[k](0) = defect;

        prob.R[k](0, 0) = 2.0 * cfg.lambda * dt;
        prob.r[k](0) = 2.0 * cfg.lambda * dt * u[k];

        if (k > 0) {
            prob.Q[k](0, 0) = 2.0;
            prob.q[k](0) = 2.0 * (y[k] - y_obs[k]);
        }
    }

    prob.P_N = Eigen::MatrixXd::Zero(1, 1);
    prob.P_N(0, 0) = 2.0;
    prob.p_N = Eigen::VectorXd::Zero(1);
    prob.p_N(0) = 2.0 * (y[cfg.N] - y_obs[cfg.N]);
    prob.gamma_N = 0.0;

    return prob;
}

static std::vector<double> eigen_to_std(const Eigen::VectorXd& v)
{
    std::vector<double> out(static_cast<std::size_t>(v.size()), 0.0);
    for (int i = 0; i < v.size(); ++i)
        out[static_cast<std::size_t>(i)] = v(i);
    return out;
}

static std::vector<double> add_scaled(const std::vector<double>& a,
                                      const std::vector<double>& b,
                                      double alpha)
{
    if (a.size() != b.size())
        throw std::runtime_error("add_scaled: size mismatch");

    std::vector<double> out = a;
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] += alpha * b[i];
    return out;
}

static FdCheckResult finite_difference_check(const ExperimentConfig& cfg,
                                             const std::vector<double>& u,
                                             const std::vector<double>& y_obs,
                                             const Eigen::VectorXd& grad,
                                             const Eigen::VectorXd& dir)
{
    const std::vector<double> d = eigen_to_std(dir);

    std::vector<double> u_plus = add_scaled(u, d, cfg.fd_eps);
    std::vector<double> u_minus = add_scaled(u, d, -cfg.fd_eps);

    const std::vector<double> y_plus = rollout_state(cfg, u_plus);
    const std::vector<double> y_minus = rollout_state(cfg, u_minus);

    const double j_plus = compute_objective(cfg, y_plus, u_plus, y_obs);
    const double j_minus = compute_objective(cfg, y_minus, u_minus, y_obs);

    FdCheckResult r;
    r.fd_dir = (j_plus - j_minus) / (2.0 * cfg.fd_eps);
    r.adj_dir = grad.dot(dir);

    const double denom = std::max(1.0, std::max(std::abs(r.fd_dir), std::abs(r.adj_dir)));
    r.rel_error = std::abs(r.fd_dir - r.adj_dir) / denom;
    return r;
}

static void write_trajectory_csv(const std::string& path,
                                 const std::vector<double>& t,
                                 const std::vector<double>& y_true,
                                 const std::vector<double>& y_obs,
                                 const std::vector<double>& y_est,
                                 const std::vector<double>& u_est)
{
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("Failed to open output file: " + path);

    out << "t,y_true,y_obs,y_est,u_est\n";
    out << std::setprecision(12);

    for (std::size_t i = 0; i < t.size(); ++i) {
        out << t[i] << ',' << y_true[i] << ',' << y_obs[i] << ',' << y_est[i] << ',';
        if (i < u_est.size())
            out << u_est[i];
        else
            out << std::numeric_limits<double>::quiet_NaN();
        out << '\n';
    }
}

static void write_optimization_csv(const std::string& path,
                                   const std::vector<int>& it,
                                   const std::vector<double>& cost,
                                   const std::vector<double>& grad_norm,
                                   const std::vector<double>& alpha,
                                   const std::vector<double>& step_norm,
                                   const std::vector<double>& fd_rel,
                                   const std::vector<double>& fd_dir,
                                   const std::vector<double>& adj_dir)
{
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("Failed to open output file: " + path);

    out << "iter,cost,grad_norm,alpha,step_norm,fd_rel_error,fd_dir_deriv,adj_dir_deriv\n";
    out << std::setprecision(12);

    for (std::size_t i = 0; i < it.size(); ++i) {
        out << it[i] << ',' << cost[i] << ',' << grad_norm[i] << ',' << alpha[i] << ','
            << step_norm[i] << ',' << fd_rel[i] << ',' << fd_dir[i] << ',' << adj_dir[i] << '\n';
    }
}

static void write_iterates_csv(const std::string& path,
                               const std::vector<double>& t,
                               const std::vector<std::vector<double>>& y_iters)
{
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("Failed to open output file: " + path);

    out << "iter,t,y_est\n";
    out << std::setprecision(12);

    for (std::size_t it = 0; it < y_iters.size(); ++it) {
        for (std::size_t k = 0; k < t.size(); ++k)
            out << it << ',' << t[k] << ',' << y_iters[it][k] << '\n';
    }
}

int main()
{
    try {
        const ExperimentConfig cfg;
        const std::vector<double> t = make_time_grid(cfg);

        std::vector<double> y_true;
        std::vector<double> y_obs;
        build_data(cfg, y_true, y_obs);

        std::vector<double> u(cfg.N, 0.0);
        std::vector<double> y = rollout_state(cfg, u);

        std::mt19937 dir_gen(cfg.seed + 7);
        std::normal_distribution<double> dir_dist(0.0, 1.0);
        Eigen::VectorXd dir(cfg.N);
        for (int k = 0; k < cfg.N; ++k)
            dir(k) = dir_dist(dir_gen);
        dir /= dir.norm();

        std::vector<int> iter_hist;
        std::vector<double> cost_hist;
        std::vector<double> grad_hist;
        std::vector<double> alpha_hist;
        std::vector<double> step_hist;
        std::vector<double> fd_rel_hist;
        std::vector<double> fd_dir_hist;
        std::vector<double> adj_dir_hist;
        std::vector<std::vector<double>> y_iter_hist;

        double cost = compute_objective(cfg, y, u, y_obs);
        Eigen::VectorXd grad = discrete_adjoint_gradient(cfg, y, u, y_obs);
        FdCheckResult fd = finite_difference_check(cfg, u, y_obs, grad, dir);

        iter_hist.push_back(0);
        cost_hist.push_back(cost);
        grad_hist.push_back(grad.norm());
        alpha_hist.push_back(0.0);
        step_hist.push_back(0.0);
        fd_rel_hist.push_back(fd.rel_error);
        fd_dir_hist.push_back(fd.fd_dir);
        adj_dir_hist.push_back(fd.adj_dir);
        y_iter_hist.push_back(y);

        std::cout << "Nonlinear ODE-regularized smoothing with Riccati-SQP\n";
        std::cout << "Data generation model: y' = -0.05 y^3 + 1.2 sin(t)\n";
        std::cout << "Smoothing model:      y' = -0.05 y^3 + u\n";
        std::cout << "Grid: N=" << cfg.N << ", dt=" << cfg.dt() << ", T=" << cfg.T << "\n";
        std::cout << "lambda=" << cfg.lambda << ", noise_sigma=" << cfg.noise_sigma
                  << ", seed=" << cfg.seed << "\n\n";

        std::cout << std::left << std::setw(6) << "iter"
                  << std::setw(16) << "cost"
                  << std::setw(14) << "grad_norm"
                  << std::setw(10) << "alpha"
                  << std::setw(14) << "step_norm"
                  << std::setw(14) << "fd_rel_err"
                  << '\n';
        std::cout << std::string(74, '-') << '\n';
        std::cout << std::setw(6) << 0
                  << std::setw(16) << std::setprecision(6) << std::fixed << cost
                  << std::setw(14) << grad.norm()
                  << std::setw(10) << 0.0
                  << std::setw(14) << 0.0
                  << std::setw(14) << fd.rel_error
                  << '\n';

        std::string stop_reason = "max iterations reached";

        for (int it = 1; it <= cfg.max_sqp_iters; ++it) {
            ELQCP qp = build_increment_qp(cfg, y, u, y_obs);
            ELQCPSol sqp_step = elqcp_sol_alloc(qp);

            Eigen::VectorXd x0_delta(1);
            x0_delta(0) = 0.0;

            elqcp_solve_riccati(qp, x0_delta, sqp_step);

            Eigen::VectorXd delta_u(cfg.N);
            for (int k = 0; k < cfg.N; ++k)
                delta_u(k) = sqp_step.u[k](0);

            const double step_norm = delta_u.norm();
            if (step_norm < 1e-12) {
                stop_reason = "SQP step norm below threshold";
                break;
            }

            const std::vector<double> du = eigen_to_std(delta_u);

            double alpha = 1.0;
            bool accepted = false;
            std::vector<double> u_trial;
            std::vector<double> y_trial;
            double trial_cost = cost;

            while (alpha >= cfg.ls_min_alpha) {
                u_trial = add_scaled(u, du, alpha);
                y_trial = rollout_state(cfg, u_trial);
                trial_cost = compute_objective(cfg, y_trial, u_trial, y_obs);

                if (trial_cost < cost) {
                    accepted = true;
                    break;
                }
                alpha *= cfg.ls_beta;
            }

            if (!accepted) {
                stop_reason = "line-search failed to find descent step";
                break;
            }

            const double old_cost = cost;
            u = u_trial;
            y = y_trial;
            cost = trial_cost;

            grad = discrete_adjoint_gradient(cfg, y, u, y_obs);
            fd = finite_difference_check(cfg, u, y_obs, grad, dir);

            iter_hist.push_back(it);
            cost_hist.push_back(cost);
            grad_hist.push_back(grad.norm());
            alpha_hist.push_back(alpha);
            step_hist.push_back(step_norm);
            fd_rel_hist.push_back(fd.rel_error);
            fd_dir_hist.push_back(fd.fd_dir);
            adj_dir_hist.push_back(fd.adj_dir);
            y_iter_hist.push_back(y);

            std::cout << std::setw(6) << it
                      << std::setw(16) << cost
                      << std::setw(14) << grad.norm()
                      << std::setw(10) << alpha
                      << std::setw(14) << step_norm
                      << std::setw(14) << fd.rel_error
                      << '\n';

            if (grad.norm() < cfg.grad_tol) {
                stop_reason = "gradient norm tolerance reached";
                break;
            }

            const double rel_improvement = std::abs(old_cost - cost) / std::max(1.0, std::abs(old_cost));
            if (rel_improvement < cfg.rel_cost_tol) {
                stop_reason = "relative cost improvement tolerance reached";
                break;
            }
        }

        const std::string traj_path = "nonlinear_smoothing_trajectory.csv";
        const std::string opt_path = "nonlinear_smoothing_optimization.csv";
        const std::string iters_path = "nonlinear_smoothing_iterates.csv";

        write_trajectory_csv(traj_path, t, y_true, y_obs, y, u);
        write_optimization_csv(opt_path, iter_hist, cost_hist, grad_hist, alpha_hist,
                               step_hist, fd_rel_hist, fd_dir_hist, adj_dir_hist);
        write_iterates_csv(iters_path, t, y_iter_hist);

        std::cout << "\nStop reason: " << stop_reason << '\n';
        std::cout << "Final cost: " << cost << '\n';
        std::cout << "Final grad norm: " << grad.norm() << '\n';
        std::cout << "\nWrote: " << traj_path << '\n';
        std::cout << "Wrote: " << opt_path << '\n';
        std::cout << "Wrote: " << iters_path << '\n';

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
