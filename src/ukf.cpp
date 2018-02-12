#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <fstream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // load debug and tuning parameters from file to avoid recompile
  ifstream config_file;
  config_file.open("../config.in");
  config_file >> use_laser_ >> use_radar_ >> use_NIS_;
  config_file >> std_a_ >> std_yawdd_;
  config_file >> print_details_;
  cout << "Process error: std_a = " << std_a_ << " std_yawdd = " << std_yawdd_ << endl;
  config_file.close();

  /*
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // do NIS calculation only when this this true
  use_NIS_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.4;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;
  */

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  
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

  is_initialized_ = false;

  previous_timestamp_ = 0;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  double weight = 0.5 / (lambda_ + n_aug_);
  for(int i = 1; i < 2 * n_aug_ + 1; ++i) weights_(i) = weight;

  radar_NIS_outliers_ = 0;
  lidar_NIS_outliers_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if(meas_package.sensor_type_ == MeasurementPackage::LASER && print_details_) {
    cout << "Laser Observed position: " << meas_package.raw_measurements_.transpose() << endl;
  }
  if(! is_initialized_) {
    // first measurement
    cout << "Init EKF: " << (meas_package.sensor_type_ == MeasurementPackage::RADAR ? "Radar" : "Lidar") << endl
         << meas_package.raw_measurements_.transpose() << endl;

    // only need to init x, P
    Eigen::VectorXd x_in(5);
    Eigen::MatrixXd P_in(5, 5);
    // wild guess
    P_in << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 10, 0, 0,
            0, 0, 0, 10, 0,
            0, 0, 0, 0, 10;

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
    cout << "Initialized!" << endl;
  }
  else { // perform a normal predict/update cycle
    double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = meas_package.timestamp_;
    Prediction(dt);
    if(use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // radar meas
      Update(meas_package);
    }
    else if(use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // lidar meas
      Update(meas_package);
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
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

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
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
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
  x_(3) = ConfinedRad(x_(3));

  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = ConfinedRad(x_diff(3));
    P_ += weights_(i) * x_diff * x_diff.transpose();
  }

  if(print_details_) {
    cout << "Prediction: " << x_.transpose() << endl;
    cout << "Uncertainty: " << endl << P_ << endl;
  }
}

/**
 * Updates the state and the state covariance matrix using a radar or laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::Update(MeasurementPackage meas_package) {
  /**
   * Predict measurement sigma points, mean, and covariance
   */
  const int n_z = (meas_package.sensor_type_ == MeasurementPackage::RADAR ? 3 : 2);
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    Zsig.col(i) = StateToMeas(Xsig_pred_.col(i), meas_package.sensor_type_);
  }

  //calculate mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred += weights_(i) * Zsig.col(i);
  }
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) z_pred(1) = ConfinedRad(z_pred(1));

  //calculate innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    S << std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0, std_radrd_ * std_radrd_;
  }
  else {
    S << std_laspx_ * std_laspx_, 0,
         0, std_laspy_ * std_laspy_;
  }

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd diff = Zsig.col(i) - z_pred;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) diff(1) = ConfinedRad(diff(1));
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
    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      x_diff(3) = ConfinedRad(x_diff(3));
      z_diff(1) = ConfinedRad(z_diff(1));
    }

    T += weights_(i) * x_diff * z_diff.transpose();
  }
  //calculate Kalman gain K;
  MatrixXd K = T * S.inverse();
  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    z_diff(1) = ConfinedRad(z_diff(1));
  }
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  if (print_details_) {
    cout << "Pred meas: " << z_pred.transpose() << endl;
    cout << "Actual meas: " << z.transpose() << endl;
    cout << "New x: " << x_.transpose() << endl;
  }

  /**
   * Calculate NIS and outlier percentage
   */
  if(use_NIS_) {
    double NIS = z_diff.transpose() * S.inverse() * z_diff;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      radar_NIS_.push_back(NIS);
      if (NIS >= 7.815) radar_NIS_outliers_++;
      cout << "Radar count: " << radar_NIS_.size()
           << ". NIS value: " << NIS
           << ". Outlier percentage: " << double(radar_NIS_outliers_) / radar_NIS_.size() << endl;
    }
    else {
      lidar_NIS_.push_back(NIS);
      if (NIS >= 5.991) lidar_NIS_outliers_++;
      cout << "Lidar count: " << lidar_NIS_.size()
           << ". NIS value: " << NIS
           << ". Outlier percentage: " << double(lidar_NIS_outliers_) / lidar_NIS_.size() << endl;
    }
  }
}

/**
 * Convert state vector to Radar or Laser measurement space vector
 * @param x state vector, px, py, v, yaw, yawd
 * @return z Radar measurement space vector, rho, theta, rho_dot
 *        or Laser measurement space vector, px, py
 */
VectorXd UKF::StateToMeas(VectorXd x, MeasurementPackage::SensorType sensor_type) {
  double px = x(0);
  double py = x(1);
  double v  = x(2);
  double yaw = x(3);
  double yawd = x(4);
  const double EPS = 0.00000001;

  VectorXd z;
  if (sensor_type == MeasurementPackage::RADAR) {
    double rho = sqrt(px * px + py * py);
    double theta;
    if (fabs(py) < EPS && fabs(px) < EPS) theta = 0;
    else theta = atan2(py, px);
    double rho_dot = (px * cos(yaw) * v + py * sin(yaw) * v) / std::max(rho, EPS);
    z = VectorXd(3);
    z << rho, theta, rho_dot;
  }
  else {
    z = VectorXd(2);
    z << px, py;
  }
  return z;
}

/**
 * Confine rad values to be between -PI and PI
 */
double UKF::ConfinedRad(double _x) {
  double x = _x;
  while (x > M_PI) x -= 2 * M_PI;
  while (x < -M_PI) x += 2 * M_PI;
  return x;
}
