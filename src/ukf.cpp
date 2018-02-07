#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // do NIS calculation only when this this true
  use_NIS_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;

  previous_timestamp_ = 0;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  double weight = 0.5 / (lambda_ + n_aug_);
  for(int i = 0; i < 2 * n_aug_ + 1; ++i) weights_(i) = weight;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if(! is_initialized_) {
    // first measurement
    cout << "Init EKF: " << (meas_package.sensor_type_ == MeasurementPackage::RADAR ? "Radar" : "Lidar") << endl
         << meas_package.raw_measurements_ << endl;

    // only need to init x, P
    Eigen::VectorXd x_in(5);
    Eigen::MatrixXd P_in(5, 5);
    // wild guess
    P_in << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1000, 0, 0,
            0, 0, 0, 10, 0,
            0, 0, 0, 0, 100;

    // init x, P with first measurement
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double ro = meas_package.raw_measurements_[0];
      double theta = meas_package.raw_measurements_[1];
      double ro_dot = meas_package.raw_measurements_[2];
      x_in << ro * cos(theta), ro * sin(theta), 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_in << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }

    x_ = x_in;
    P_ = P_in;

    // done initializing, no need to predict or update
    previous_timestamp_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }
  else { // perform a normal predict/update cycle
    double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
    Prediction(dt);
    if(use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // radar meas
      UpdateRadar(meas_package);
    }
    else if(use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // lidar meas
      UpdateLidar(meas_package);
    }
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
   * Generate augmented state and sigma points
   */
  // augmented mean state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug.tail(2).setZero();

  // augmented covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();

  //create augmented sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  double coeff = sqrt(n_aug_ + lambda_);
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i){
    Xsig_aug.col(1 + i) = x_aug + coeff * A_aug.col(i);
    Xsig_aug.col(n_aug_ + 1 + i) = x_aug - coeff * A_aug.col(i);
  }

  /**
   * Predict augmented sigma points
   */
  for (int i = 0; i < 2 * n_aug_ + 1; ++i){
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    double pxp, pyp;

    if(fabs(yawd) > 0.000001) {
      pxp = px + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      pyp = py + v / yawd * (-cos(yaw + yawd * delta_t) + cos(yaw));
    }
    else{
      pxp = px + v * (cos(yaw)) * delta_t;
      pyp = py + v * (sin(yaw)) * delta_t;
    }
    pxp += 0.5 * delta_t * delta_t * cos(yaw) * nu_a;
    pyp += 0.5 * delta_t * delta_t * sin(yaw) * nu_a;

    double vp = v + delta_t * nu_a;
    double yawp = yaw + yawd * delta_t + 0.5 * delta_t * delta_t * nu_yawdd;
    double yawdp = yawd + delta_t * nu_yawdd;

    Xsig_pred_(0, i) = pxp;
    Xsig_pred_(1, i) = pyp;
    Xsig_pred_(2, i) = vp;
    Xsig_pred_(3, i) = yawp;
    Xsig_pred_(4, i) = yawdp;
  }

  /**
   * Calculate the mean state and covariance matrix
   */
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    P_ += weights_(i) * (Xsig_pred_.col(i) - x_) * (Xsig_pred_.col(i) - x_).transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  UpdateCommon(meas_package);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * Predict measurement sigma points, mean, and covariance
   */
  const int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    Zsig.col(i) = StateToRadarMeas(Xsig_pred_.col(i));
  }

  //calculate mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred += weights_(i) * Zsig.col(i);
  }
  //calculate innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S << std_radr_ * std_radr_, 0, 0,
      0, std_radphi_ * std_radphi_, 0,
      0, 0, std_radrd_ * std_radrd_;

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd diff = Zsig.col(i) - z_pred;
    while(diff(1) > M_PI) diff(1) -= 2 * M_PI;
    while(diff(1) < - M_PI) diff(1) += 2 * M_PI;
    S += weights_(i) * diff * diff.transpose();
  }

  /**
   * Update x and P
   */
  VectorXd z = meas_package.raw_measurements_;
  //calculate cross correlation matrix
  MatrixXd T = MatrixXd(n_x_, n_z);
  T.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) > M_PI) x_diff(3) -= 2 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2 * M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) > M_PI) z_diff(1) -= 2 * M_PI;
    while (z_diff(1) < - M_PI) z_diff(1) += 2 * M_PI;
    T += weights_(i) * x_diff * z_diff.transpose();
  }
  //calculate Kalman gain K;
  MatrixXd K = T * S.inverse();
  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  while (z_diff(1) > M_PI) z_diff(1) -= 2 * M_PI;
  while (z_diff(1) < - M_PI) z_diff(1) += 2 * M_PI;
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  /**
   * Calculate NIS and outlier percentage
   */
  if(use_NIS_) {
    double NIS = z_diff.transpose() * S.inverse() * z_diff;
    radar_NIS_.push_back(NIS);
    int count = radar_NIS_.size();
    int outlier_count = 0;
    for (double it : radar_NIS_) outlier_count += (it >= 7.815 ? 1 : 0);
    cout << "Radar count: " << count << ". Outlier percentage:" << double(outlier_count) / count << endl;
  }
}

void UKF::UpdateCommon(MeasurementPackage meas_package) {
  // pass
}

VectorXd UKF::StateToLidarMeas(VectorXd x) {
  VectorXd z = VectorXd(2);
  z = x.head(2);
  return z;
}

VectorXd UKF::StateToRadarMeas(VectorXd x) {
  double px = x(0);
  double py = x(1);
  double v  = x(2);
  double yaw = x(3);
  double yawd = x(4);

  double rho = sqrt(px * px + py * py);
  double theta = atan2(py, px);
  double rho_dot = (px * cos(yaw) * v + py * sin(yaw) * v) / rho;
  VectorXd z = VectorXd(3);
  z << rho, theta, rho_dot;
  return z;
}