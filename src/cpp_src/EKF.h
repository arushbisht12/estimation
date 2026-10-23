#pragma once

#include <Eigen/Dense>
#include <cmath>


static const int NUM_VARS = 10;

using Vector11d = Eigen::Vector<double, 11>;
using Vector10d = Eigen::Vector<double, NUM_VARS>;
using Vector4d = Eigen::Vector<double, 4>;
using Vector3d = Eigen::Vector<double, 3>;
using Vector1d = Eigen::Vector<double, 1>;

using Matrix10d = Eigen::Matrix<double, NUM_VARS, NUM_VARS>;

struct State
{
    Vector4d q;
    Vector3d v;
    Vector3d p;

    Vector10d asVector() const {
        Vector10d x;
        x << q, v, p;
        return x;
    };

    State fromVector(const Vector10d x) {
        State state;
        state.q = x.segment<4>(0);
        state.v = x.segment<3>(4);
        state.p = x.segment<3>(7);
        return state;
    }
};



class EKF
{
  public:
    static constexpr double g_const = 9.80665;
    inline static const Vector3d g_vector{0, 0, g_const};
    static constexpr double theta_const = 0;
    inline static const Vector3d north_vector{0, std::cos(theta_const), std::sin(theta_const)};


    EKF(Vector4d initial_q, Vector3d initial_vel, Vector3d initial_pos)
    {
        Vector10d tmpx;
        tmpx << initial_q, initial_vel, initial_pos;
        x = x.fromVector(tmpx);

        P = Eigen::Matrix<double,10,10>::Identity();

    }

    void prediction(const double& dt,
                    const Vector3d gyr = Vector3d::Zero(),
                    const Vector3d acc = Vector3d::Zero())
    {
        set_x(f(gyr, acc, dt));

        Eigen::Matrix<double,10,6> W_k = W(dt);
        Matrix10d Q = W_k * process_noise * W_k.transpose();

        Matrix10d F_k = F(gyr, acc, dt);
        P = F_k * P * F_k.transpose() + Q;
    }


    void correction(const double& dt,
                    Vector3d acc = Vector3d::Zero(),
                    Vector3d mag = Vector3d::Zero(),
                    Vector3d gps = Vector3d::Zero(),
                    Vector1d lid = Vector1d::Zero(),
                    Vector1d bar = Vector1d::Zero())
    {
        Vector3d acc_norm = acc / acc.norm();
        Vector3d mag_norm = mag / mag.norm();

        Eigen::Matrix<double,11,10> H_k = H(theta_const);
        Eigen::Matrix<double,11,11> S = H_k * P * H_k.transpose() + R;
        // find out what type of matrix S is, what decomposition is best
        Eigen::Matrix<double,10,11> K = P * H_k.transpose() * S.colPivHouseholderQr().solve(Eigen::Matrix<double,11,11>::Identity());

        Vector11d z;
        z << acc_norm, mag_norm, gps, lid, bar;
        set_x(x.asVector() + K * (z - h(dt)));

        P = (Eigen::Matrix<double,10,10>::Identity() - K * H_k) * P;
    }



    Vector10d f(const Vector3d& gyr, const Vector3d& acc, const double& dt) {
        const Vector4d q_prev = x.q;
        const Vector3d vel_prev = x.v;
        const Vector3d pos_prev = x.p;

        Vector4d q_new = (Eigen::Matrix<double,4,4>::Identity() + 0.5 * dt * omega(gyr)) * q_prev;
        Vector3d vel_new = vel_prev + dt * rotation_body_to_earth(q_new) * acc;
        vel_new(2) -= g_const * dt;
        Vector3d pos_new = pos_prev + vel_new * dt;

        Vector10d x_new;
        x_new << q_new, vel_new, pos_new;
        return x_new;
    }

    Vector11d h(const double& dt) {
        Vector3d acc_ex = rotation_earth_to_body(x.q) * g_vector;
        Vector3d mag_ex = rotation_earth_to_body(x.q) * north_vector;
        Vector3d gps_ex = x.p;
        Vector1d lid_ex;
        lid_ex << x.p(2);
        Vector1d bar_ex;
        bar_ex << x.p(2);

        Vector11d ex;
        ex << acc_ex, mag_ex, gps_ex, lid_ex, bar_ex;
        return ex;
    }

    Matrix10d F(const Vector3d& gyr, const Vector3d& acc, const double& dt) {
        const double w1 = gyr(0);
        const double w2 = gyr(1);
        const double w3 = gyr(2);

        const double a1 = acc(0);
        const double a2 = acc(1);
        const double a3 = acc(2);

        const double q1 = x.q(0);
        const double q2 = x.q(1);
        const double q3 = x.q(2);
        const double q4 = x.q(3);

        Matrix10d F_sym;

        F_sym <<  1,        -(dt*w1)/2,  -(dt*w2)/2,  -(dt*w3)/2,  0, 0, 0, 0, 0, 0,
            (dt*w1)/2, 1,           (dt*w2)/2,  -(dt*w3)/2,  0, 0, 0, 0, 0, 0,
            (dt*w2)/2, -(dt*w3)/2,  1,           (dt*w1)/2,  0, 0, 0, 0, 0, 0,
            (dt*w3)/2,  (dt*w2)/2, -(dt*w1)/2,  1,           0, 0, 0, 0, 0, 0,

            dt*(2*a1*q1 - 2*a2*q4 + 2*a3*q3),
            dt*(2*a1*q2 + 2*a2*q3 + 2*a3*q4),
            dt*(2*a2*q2 - 2*a1*q3 + 2*a3*q1),
            -dt*(2*a2*q1 + 2*a1*q4 - 2*a3*q2),
            1, 0, 0, 0, 0, 0,

            dt*(2*a2*q1 + a1*q4 - 2*a3*q2),
            -dt*(2*a2*q2 - a1*q3 + 2*a3*q1),
            dt*(a1*q2 + 2*a2*q3 + 2*a3*q4),
            dt*(a1*q1 - 2*a2*q4 + 2*a3*q3),
            0, 1, 0, 0, 0, 0,

            dt*(2*a2*q2 - 2*a1*q3 + 2*a3*q1),
            dt*(2*a2*q1 + 2*a1*q4 - 2*a3*q2),
            -dt*(2*a1*q1 - 2*a2*q4 + 2*a3*q3),
            dt*(2*a1*q2 + 2*a2*q3 + 2*a3*q4),
            0, 0, 1, 0, 0, 0,

            0, 0, 0, 0, dt, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, dt, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, dt, 0, 0, 1;

        return F_sym;
    };

    Eigen::Matrix<double,11,10> H(const double& theta) {
        const double q1 = x.q(0);
        const double q2 = x.q(1);
        const double q3 = x.q(2);
        const double q4 = x.q(3);

        const double c = std::cos(theta);
        const double s = std::sin(theta);

        Eigen::Matrix<double, 11, 10> H_sym;

        H_sym <<  2*g_const*q3, -2*g_const*q4,  2*g_const*q1, -2*g_const*q2, 0, 0, 0, 0, 0, 0,
            -2*g_const*q2, -2*g_const*q1, -2*g_const*q4, -2*g_const*q3, 0, 0, 0, 0, 0, 0,
            -2*g_const*q1,  2*g_const*q2,  2*g_const*q3, -2*g_const*q4, 0, 0, 0, 0, 0, 0,

            q4*c + 2*q3*s,  q3*c - 2*q4*s,  q2*c + 2*q1*s,  q1*c - 2*q2*s, 0,0,0,0,0,0,
            2*q1*c - 2*q2*s, -2*q2*c - 2*q1*s, 2*q3*c - 2*q4*s, -2*q4*c - 2*q3*s, 0,0,0,0,0,0,
            -2*q2*c - 2*q1*s, 2*q2*s - 2*q1*c, 2*q4*c + 2*q3*s,  2*q3*c - 2*q4*s, 0,0,0,0,0,0,

            0,0,0,0, 0,0,0,1,0,0,
            0,0,0,0, 0,0,0,0,1,0,
            0,0,0,0, 0,0,0,0,0,1,
            0,0,0,0, 0,0,0,0,0,1,
            0,0,0,0, 0,0,0,0,0,1;

        return H_sym;
    }

    Eigen::Matrix<double,10,6> W(const double& dt) {
        const double q1 = x.q(0);
        const double q2 = x.q(1);
        const double q3 = x.q(2);
        const double q4 = x.q(3);

        Eigen::Matrix<double, 10, 6> W_sym;

        W_sym <<  -(dt*q2)/2, -(dt*q3)/2, -(dt*q4)/2, 0, 0, 0,
            (dt*q1)/2,  (dt*q3)/2, -(dt*q4)/2, 0, 0, 0,
            (dt*q4)/2,  (dt*q1)/2, -(dt*q2)/2, 0, 0, 0,
            -(dt*q3)/2,  (dt*q2)/2,  (dt*q1)/2, 0, 0, 0,

            0,0,0, dt*(q1*q1 + q2*q2 - q3*q3 - q4*q4), -dt*(2*q1*q4 - 2*q2*q3), dt*(2*q1*q3 + 2*q2*q4),
            0,0,0, dt*(q1*q4 + q2*q3), dt*(q1*q1 - q2*q2 + q3*q3 - q4*q4), -dt*(2*q1*q2 - 2*q3*q4),
            0,0,0, -dt*(2*q1*q3 - 2*q2*q4), dt*(2*q1*q2 + 2*q3*q4), dt*(q1*q1 - q2*q2 - q3*q3 + q4*q4),

            0,0,0,0,0,0,
            0,0,0,0,0,0,
            0,0,0,0,0,0;

        return W_sym;
    }

    Eigen::Matrix<double,4,4> omega(const Vector3d& gyr) {
        const double gp = gyr(0);
        const double gq = gyr(1);
        const double gr = gyr(2);

        Eigen::Matrix<double,4,4> Omega;

        Omega <<  0.0, -gp, -gq, -gr,
            gp,  0.0,  gr, -gq,
            gq, -gr,  0.0,  gp,
            gr,  gq, -gp,  0.0;

        return Omega;
    }

    Eigen::Matrix<double,3,3> rotation_body_to_earth(const Vector4d& q) {
        const double a = q(0);
        const double b = q(1);
        const double c = q(2);
        const double d = q(3);

        Eigen::Matrix<double,3,3> R;

        R <<  a*a + b*b - c*c - d*d,   -2*a*d + 2*b*c,        2*a*c + 2*b*d,
            2*a*d + 2*b*c,           a*a - b*b + c*c - d*d, -2*a*b + 2*c*d,
            -2*a*c + 2*b*d,           2*a*b + 2*c*d,         a*a - b*b - c*c + d*d;

        return R;
    }

    Eigen::Matrix<double,3,3> rotation_earth_to_body(const Vector4d& q) {
        return rotation_body_to_earth(q).transpose();
    }

// getters and setters

    Vector10d mean() const {
        return x.asVector();
    }

    Matrix10d cov() const {
        return Matrix10d::Identity();
    }

    Vector4d q() {
        return x.q;
    }

    Vector3d vel() {
        return x.v;
    }

    Vector3d pos() {
        return x.p;
    }

    void set_x(Vector10d new_x) {
        x = x.fromVector(new_x);
        x.q = x.q / x.q.norm();
    }
  private:
    State x;
    Eigen::Matrix<double,10,10> P;
    const Eigen::Matrix<double,6,6> process_noise = Eigen::Matrix<double,6,6>::Zero();
    //measurement_noise
    const Eigen::Matrix<double,11,11> R = Eigen::Matrix<double,11,11>::Identity();

};
