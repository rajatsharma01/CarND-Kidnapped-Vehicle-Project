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
#include <random>
#include <sstream>
#include <string>
#include <iterator>
#include <cassert>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;

    std::default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;

        particles.push_back(p);
        weights.push_back(p.weight);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    std::default_random_engine gen;

    for (int i = 0; i < num_particles; i++) {
        // predict new states for the particle
        double x_pred, y_pred, theta_pred;
        if (yaw_rate == 0) {
            theta_pred = particles[i].theta;
            x_pred = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            y_pred = particles[i].y + velocity * delta_t * sin(particles[i].theta);
        } else {
            theta_pred = particles[i].theta + yaw_rate * delta_t;
            x_pred = particles[i].x + velocity/yaw_rate * (sin(theta_pred) - sin(particles[i].theta));
            y_pred = particles[i].y - velocity/yaw_rate * (cos(theta_pred) - cos(particles[i].theta));
        }

        // Setup particle states with Gaussian noise added
        normal_distribution<double> dist_x(x_pred, std_pos[0]);
        normal_distribution<double> dist_y(y_pred, std_pos[1]);
        normal_distribution<double> dist_theta(theta_pred, std_pos[2]);
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (auto& obs : observations) {
        double min_distance = std::numeric_limits<double>::max();
        for (auto pred : predicted) {
            double obs_distance = dist(obs.x, obs.y, pred.x, pred.y);
            if (obs_distance < min_distance) {
                obs.id = pred.id;
                min_distance = obs_distance;
            }
        }
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

    const double normalizer = 1/(2 * M_PI * std_landmark[0] * std_landmark[1]);
    for (int i = 0; i < num_particles; i++) {
        // Transform the observations into map coordinate system
        std::vector<LandmarkObs> trans_observations;
        for (auto obs : observations) {
            LandmarkObs trans_obs;
            trans_obs.id = 0; // No association to landmark yet
            trans_obs.x = particles[i].x + obs.x * cos(particles[i].theta)
                                         - obs.y * sin(particles[i].theta);
            trans_obs.y = particles[i].y + obs.x * sin(particles[i].theta)
                                         + obs.y * cos(particles[i].theta);
            trans_observations.push_back(trans_obs);
        }

        // Find lanmarks within sensor range of current particle
        std::vector<LandmarkObs> predictions;
        for (auto lm : map_landmarks.landmark_list) {
            LandmarkObs pred_lm;
            double distance = dist(lm.x_f, lm.y_f, particles[i].x, particles[i].y);
            if (distance <= sensor_range) {
                pred_lm.id = lm.id_i;
                pred_lm.x = lm.x_f;
                pred_lm.y = lm.y_f;
                predictions.push_back(pred_lm);
            }
        }

        // Associate landmark to each observation
        dataAssociation(predictions, trans_observations);

        // Debugging information about particle's association with mapped landmarks
        std::vector<int> associations;
        std::vector<double> sense_x;
        std::vector<double> sense_y;

        // Calculate new weight of particle based on Multivariate Gaussian probability of all transformed
        // observation being corresponding associated landmark. As a result of this step, particles closer
        // to true position of car would have higher weights as observations would match with map locations
        // of landmark, whereas false particles would have lesser weight for larger errors between observed
        // landmark position and map position of landmark. Later during resampling, true particles would win
        // and false particle positions would fall off the cliff.
        particles[i].weight = 1.0;
        for (auto trans_obs : trans_observations) {
            int association = trans_obs.id;
            if (association != 0) {
                assert(map_landmarks.landmark_list[association - 1].id_i == association);
                double mu_x = map_landmarks.landmark_list[association - 1].x_f;
                double mu_y = map_landmarks.landmark_list[association - 1].y_f;
                double multiplier = normalizer * exp(-pow((trans_obs.x - mu_x)/std_landmark[0], 2)
                                                     -pow((trans_obs.y - mu_y)/std_landmark[1], 2));
                if (multiplier > 0) {
                    particles[i].weight *= multiplier;
                }
                associations.push_back(association);
                sense_x.push_back(trans_obs.x);
                sense_y.push_back(trans_obs.y);
            }
        }

        // Update particle's associations for debug info
        particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
        
        // Update global weights list for next resampling step
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::default_random_engine generator;
    std::discrete_distribution<int> distribution(weights.begin(), weights.end());

    std::vector<Particle> resample_particles;

    for (int i = 0; i < num_particles; i++) {
        resample_particles.push_back(particles[distribution(generator)]);
    }

    particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
