#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using std::cout;
using std::endl;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_ << 0.1, 0.1, 0.1, 0.1, 0.1;

  // initial covariance matrix
  P_ = MatrixXd::Zero(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.7;

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

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  n_x_ = 5;

  n_aug_ = 7;

  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_x_ + 1);

  previous_timestamp_ = 0;

  NIS_radar_ = 0;

  NIS_laser_ = 0;

  n_z_radar_ = 3;

  n_z_laser_ = 2;

  int lambda = 3 - n_aug_;
  //set vector for weights
  weights_ = VectorXd(2*n_aug_+1);
  double weight_0 = lambda/(lambda+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda);
    weights_(i) = weight;
  }

  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_*std_laspx_, 0,
       0, std_laspy_*std_laspy_;

  R_radar_ = MatrixXd(n_z_radar_,n_z_radar_);
  R_radar_ <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
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
  if (!is_initialized_) {
    float px = 0;
    float py = 0;
    if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      float ro = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      if (ro < 0.001)
        ro = 0.001;
      if (phi < 0.001)
        phi = 0.001;
      px = ro * cos(phi);
      py = ro * sin(phi);
      if (px < 0.001) {
        px = 0.001;
      }

      if (py < 0.001) {
        py = 0.001;
      }
      float ro_dot = meas_package.raw_measurements_(2);

      if (ro_dot < 0.001) {
        ro_dot = 0.001;
      }
      x_ << px, py, ro_dot, phi, 0;
      previous_timestamp_ = meas_package.timestamp_;
    } else if(use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
      px = meas_package.raw_measurements_(0);
      py = meas_package.raw_measurements_(1);

      if (px < 0.001) {
        px = 0.001;
      }

      if (py < 0.001) {
        py = 0.001;
      }
      x_ << px, py, 0, 0, 0;
      previous_timestamp_ = meas_package.timestamp_;
    }

          P_ << 50, 0, 0, 0,0,
0, 50, 0, 0,0,
0, 0, 1, 0,0,
0, 0, 0, 0.5,0,
0, 0, 0, 0,0.5;

    is_initialized_ = true;
    return;
  }

  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;   //dt - expressed in seconds
  previous_timestamp_ = meas_package.timestamp_;

  if (dt > 0.001) {
    // while (dt > 0.1) {
    //   const double delta = 0.05;
    //   Prediction(delta);
    //   dt -= delta;
    // }
    Prediction(dt);
    if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      VectorXd z;
      MatrixXd S;
      MatrixXd Zsig;
      
      PredictRadarMeasurement(&z, &S, &Zsig);
      UpdateRadarState(z, S, Zsig, meas_package);
    } else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
      VectorXd z;
      MatrixXd S;
      MatrixXd Zsig;

      PredictLaserMeasurement(&z, &S, &Zsig);
      UpdateLaserState(z, S, Zsig, meas_package);
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
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  GenerateSigmaPoints();
  AugmentedSigmaPoints();
  SigmaPointPrediction(delta_t);
  PredictMeanAndCovariance();
}

void UKF::GenerateSigmaPoints() {

  int lambda = 3 - n_x_;

  MatrixXd Xsig = MatrixXd::Zero(n_x_, 2 * n_x_ + 1);

  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  //set first column of sigma point matrix
  Xsig.col(0)  = x_;

  //set remaining sigma points
  for (int i = 0; i < n_x_; i++)
  {
    Xsig.col(i+1)     = x_ + sqrt(lambda+n_x_) * A.col(i);
    Xsig.col(i+1+n_x_) = x_ - sqrt(lambda+n_x_) * A.col(i);
  }

  Xsig_pred_ = Xsig;
}

void UKF::AugmentedSigmaPoints() {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(7, 7);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

  int lambda = 3 - n_aug_;
 
  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda+n_aug_) * L.col(i);
  }
  
  Xsig_pred_ = Xsig_aug;
}

void UKF::SigmaPointPrediction(double delta_t) {

  MatrixXd Xsig_aug = Xsig_pred_;

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    yaw = constrainAngle(yaw);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }

  //write result
  Xsig_pred_ = Xsig_pred;

}

void UKF::PredictMeanAndCovariance() {
  
  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd::Zero(n_x_, n_x_);

  double lambda = 3 - n_aug_;

  //predicted state mean
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x = x+ weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    //angle normalization
    x_diff(3) = constrainAngle(x_diff(3));

    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }

  //write result
  x_ = x;
  P_ = P;
}

void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out) {

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z_radar_, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    yaw = constrainAngle(yaw);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_radar_);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z_radar_,n_z_radar_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    z_diff(1) = constrainAngle(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R_radar_;

  //write result
  *Zsig_out = Zsig;
  *z_out = z_pred;
  *S_out = S;
}

void UKF::UpdateRadarState(VectorXd z_pred, MatrixXd S, MatrixXd Zsig, MeasurementPackage meas_package) {

  //create example vector for incoming radar measurement
  VectorXd z = meas_package.raw_measurements_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_radar_);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    z_diff(1) = constrainAngle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = constrainAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd Si = S.inverse();
  MatrixXd K = Tc * Si;

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  z_diff(1) = constrainAngle(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  NIS_radar_ = z_diff.transpose() * Si * z_diff;
}

void UKF::PredictLaserMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out) {

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z_laser_, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    yaw = constrainAngle(yaw);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = p_x;                        //px
    Zsig(1,i) = p_y;                        //py
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_laser_);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z_laser_,n_z_laser_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R_laser_;

  //write result
  *Zsig_out = Zsig;
  *z_out = z_pred;
  *S_out = S;
}

void UKF::UpdateLaserState(VectorXd z_pred, MatrixXd S, MatrixXd Zsig, MeasurementPackage meas_package) {

  //create example vector for incoming radar measurement
  VectorXd z = meas_package.raw_measurements_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_laser_);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = constrainAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd Si = S.inverse();
  MatrixXd K = Tc * Si;

  //residual
  VectorXd z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  NIS_laser_ = z_diff.transpose() * Si * z_diff;
}


double UKF::constrainAngle(double x) {
  if (x > M_PI)
    x = fmod(x - M_PI, 2*M_PI) - M_PI;
  if (x < -M_PI)
    x = fmod(x + M_PI,  2*M_PI) + M_PI;
  return x;
}
