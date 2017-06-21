/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"

using namespace std;

inline double bivariate_normal(double x, double y, double mu_x, double mu_y,
                               double sig_x, double sig_y) {
  return exp(-((x - mu_x) * (x - mu_x) / (2. * sig_x * sig_x) +
               (y - mu_y) * (y - mu_y) / (2. * sig_y * sig_y))) /
         (2. * M_PI * sig_x * sig_y);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first
  // position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and
  // others in this file).
  num_particles = 100;

  particles.resize(num_particles);

  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  std::default_random_engine gen;

  for (int i = 0; i < num_particles; i++) {
    Particle &p = particles[i];

    double x_pred;
    double y_pred;
    double theta_pred;

    theta_pred = p.theta + yaw_rate * delta_t;
    x_pred = p.x + velocity / yaw_rate * (sin(theta_pred) - sin(p.theta));
    y_pred = p.y - velocity / yaw_rate * (cos(theta_pred) - cos(p.theta));

    if (fabs(yaw_rate) < 1e-6) {
      x_pred = p.x + (velocity * delta_t * cos(p.theta));
      y_pred = p.y + (velocity * delta_t * sin(p.theta));
      theta_pred = p.theta + yaw_rate * delta_t;
    } else {
      x_pred = p.x +
               (velocity / yaw_rate) *
                   (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      y_pred = p.y +
               (velocity / yaw_rate) *
                   (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
      theta_pred = p.theta + yaw_rate * delta_t;
    }

    std::normal_distribution<double> dist_x(x_pred, std_pos[0]);
    std::normal_distribution<double> dist_y(y_pred, std_pos[1]);
    std::normal_distribution<double> dist_theta(theta_pred, std_pos[2]);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement that is closest to each observed
  // measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will
  // probably find it useful to
  //   implement this method and use it as a helper during the updateWeights
  //   phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read
  //   more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located
  //   according to the MAP'S coordinate system. You will need to transform
  //   between the two systems.
  //   Keep in mind that this transformation requires both rotation AND
  //   translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement
  //   (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  for (int p_i = 0; p_i < num_particles; p_i++) {
    Particle &p = particles[p_i];

    std::vector<LandmarkObs> obs_in_map_coord;

    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs ob = observations[j];

      if (dist(observations[j].x, observations[j].y, 0, 0) <= sensor_range) {
        LandmarkObs ob_in_map;
        double theta = -p.theta;
        ob_in_map.x = ob.x * cos(theta) + ob.y * sin(theta);
        ob_in_map.y = -ob.x * sin(theta) + ob.y * cos(theta);

        ob_in_map.x += p.x;
        ob_in_map.y += p.y;

        obs_in_map_coord.push_back(ob_in_map);
      }
    }

    std::vector<LandmarkObs> map_landmarks_obs;

    for (int i = 0; i < map_landmarks.landmark_list.size(); i++) {
      LandmarkObs map_ob;
      map_ob.x = map_landmarks.landmark_list[i].x_f;
      map_ob.y = map_landmarks.landmark_list[i].y_f;
      map_ob.id = map_landmarks.landmark_list[i].id_i;
      map_landmarks_obs.push_back(map_ob);
    }

    // calculate the weight for this particle
    double wg = 1.;
    for (int i = 0; i < obs_in_map_coord.size(); i++) {
      double min_dist = 1e100;
      int min_index = -1;

      for (int j = 0; j < map_landmarks_obs.size(); j++) {
        double dst = dist(obs_in_map_coord[i].x, obs_in_map_coord[i].y,
                          map_landmarks_obs[j].x, map_landmarks_obs[j].y);
        if (dst < min_dist) {
          min_dist = dst;
          min_index = j;
        }
      }

      wg *= bivariate_normal(obs_in_map_coord[i].x, obs_in_map_coord[i].y,
                             map_landmarks_obs[min_index].x,
                             map_landmarks_obs[min_index].y, std_landmark[0],
                             std_landmark[1]);
    }
    p.weight = wg;
  }

  // normalize weights
  double wg_sum = 0.;
  for (int i = 0; i < num_particles; i++) {
    wg_sum += particles[i].weight;
  }

  for (int i = 0; i < num_particles; i++) {
    particles[i].weight /= wg_sum;
  }
}

void ParticleFilter::resample() {
  // TODO: 	 particles with replacement with probability proportional to
  // their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::default_random_engine gen;

  std::vector<double> weights(particles.size());
  for (int i = 0; i < particles.size(); i++) {
    weights[i] = particles[i].weight;
  }

  std::discrete_distribution<> distribution(weights.begin(), weights.end());
  std::vector<Particle> new_particles;

  for (int i = 0; i < num_particles; i++) {
    int weighted_index = distribution(gen);
    new_particles.push_back(particles[weighted_index]);
  }

  particles = new_particles;

  for (int i = 0; i < num_particles; i++) {
    particles[i].weight = 1.;
  }
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
