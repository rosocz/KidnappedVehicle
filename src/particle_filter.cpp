/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  default_random_engine gen;

  normal_distribution<double> distX(x, std[0]);
  normal_distribution<double> distY(y, std[1]);
  normal_distribution<double> distTheta(theta, std[2]);

  num_particles = 200;
  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = distX(gen);
    particle.y = distY(gen);
    particle.theta = distTheta(gen);
    particle.weight = 1;
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;
  double predictionX;
  double predictionY;
  double predictionTheta;

  for (unsigned int i = 0; i < particles.size(); i++) {
    Particle particle = particles[i];

    if (fabs(yaw_rate) > EPS) {
      predictionX = particle.x + (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
      predictionY = particle.y + (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
      predictionTheta = particle.theta + (yaw_rate * delta_t);
    } else {
      predictionX = particle.x + velocity * delta_t * cos(particle.theta);
      predictionY = particle.y + velocity * delta_t * sin(particle.theta);
      predictionTheta = particle.theta;
    }

    normal_distribution<double> xDistribution(predictionX, std_pos[0]);
    normal_distribution<double> yDistribution(predictionY, std_pos[1]);
    normal_distribution<double> thetaDistribution(predictionTheta, std_pos[2]);

    particle.x = xDistribution(gen);
    particle.y = yDistribution(gen);
    particle.theta = thetaDistribution(gen);

    particles[i] = particle;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  for (unsigned int i = 0; i < observations.size(); i++) {
    LandmarkObs observation = observations[i];
    double minimalDistance = numeric_limits<double>::max();

    for (unsigned int j = 0; j < predicted.size(); j++) {
      LandmarkObs prediction = predicted[j];

      double tempDistance = dist(prediction.x, prediction.y, observation.x, observation.y);
      if (tempDistance < minimalDistance) {
        minimalDistance = tempDistance;
        observation.id = prediction.id;
      }
    }

    observations[i] = observation;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  double weightNormalizer = 0.0;
  double multivariateProbability;

  for (int i = 0; i < num_particles; i++) {
    Particle particle = particles[i];
    vector<LandmarkObs> transformedObservations;

    for (unsigned int j = 0; j < observations.size(); j++) {
      LandmarkObs transformed_obs;
      transformed_obs.id = j;
      transformed_obs.x =
              particle.x + (cos(particle.theta) * observations[j].x) - (sin(particle.theta) * observations[j].y);
      transformed_obs.y =
              particle.y + (sin(particle.theta) * observations[j].x) + (cos(particle.theta) * observations[j].y);
      transformedObservations.push_back(transformed_obs);
    }

    vector<LandmarkObs> predictedLandmarks;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      Map::single_landmark_s currentLandmark = map_landmarks.landmark_list[j];
      if ((fabs((particle.x - currentLandmark.x_f)) <= sensor_range) &&
          (fabs((particle.y - currentLandmark.y_f)) <= sensor_range)) {
        predictedLandmarks.push_back(LandmarkObs{currentLandmark.id_i, currentLandmark.x_f, currentLandmark.y_f});
      }
    }

    dataAssociation(predictedLandmarks, transformedObservations);

    particle.weight = 1.0;

    double sigma_x_2 = pow(std_landmark[0], 2);
    double sigma_y_2 = pow(std_landmark[1], 2);
    double normalizer = (1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]));

    for (unsigned int k = 0; k < transformedObservations.size(); k++) {
      double trans_obs_x = transformedObservations[k].x;
      double trans_obs_y = transformedObservations[k].y;
      double trans_obs_id = transformedObservations[k].id;

      for (unsigned int l = 0; l < predictedLandmarks.size(); l++) {
        if (trans_obs_id == predictedLandmarks[l].id) {
          multivariateProbability = normalizer * exp(-1.0 * ((pow((trans_obs_x - predictedLandmarks[l].x), 2) / (2.0 * sigma_x_2)) +
                                                (pow((trans_obs_y - predictedLandmarks[l].y), 2) / (2.0 * sigma_y_2))));
          particle.weight *= multivariateProbability;
        }
      }
    }
    weightNormalizer += particle.weight;

    particles[i] = particle;
  }

  for (unsigned int i = 0; i < particles.size(); i++) {
    particles[i].weight /= weightNormalizer;
    weights[i] = particles[i].weight;
  }

}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<Particle> resampledParticles;

  default_random_engine gen;

  std::discrete_distribution<> d(weights.begin(), weights.end());

  for (unsigned int i = 0; i < particles.size(); i++) {
    int index = d(gen);
    Particle particle = particles[index];
    resampledParticles.push_back(particle);
  }

  particles = resampledParticles;

}

void ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
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
