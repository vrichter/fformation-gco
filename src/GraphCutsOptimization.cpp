// author: Viktor Richter vrichter@techfak.uni-bielefeld.de
// licence: MIT

#include "GraphCutsOptimization.h"
#include "../gco/src/GCoptimization.h"
#include <algorithm>
#include <random>

using fformation::GroupDetectorFactory;
using fformation::GroupDetector;
using fformation::Options;
using fformation::Option;
using fformation::Settings;
using fformation::Observation;
using fformation::Classification;
using fformation::IdGroup;
using fformation::Person;
using fformation::PersonId;
using fformation::Position2D;
using fformation::Timestamp;

GraphCutsOptimization::GraphCutsOptimization(const fformation::Options options)
    : GroupDetector(options),
      _mdl(options.getValue<double>("mdl",
                                    fformation::validators::Min<double>(0.))),
      _stride(options.getValue<double>(
          "stride", fformation::validators::Min<double>(0.))),
      _shuffle(options.getValueOr<bool>("shuffle",false)) {}

GroupDetectorFactory::ConstructorFunction GraphCutsOptimization::creator() {
  return [](const Options &options) {
    return GroupDetector::Ptr(new GraphCutsOptimization(options));
  };
}

Classification
GraphCutsOptimization::init(const Observation observation) const {
  fformation::NonGroupDetector init;
  return init.detect(observation);
}

std::vector<Position2D>
GraphCutsOptimization::updateGroupCenters(const Observation &obs,
                                          const Classification &cl,
                                          const Person::Stride &stride) const {
  std::vector<Position2D> group_centers;
  for (auto group : cl.createGroups(obs)) {
    group_centers.push_back(group.calculateCenter(stride));
  }
  return group_centers;
}

void calculateCosts(const Person &person, const size_t person_id,
                    const std::vector<Person> &other,
                    const std::vector<Position2D> group_centers,
                    const Person::Stride &stride,
                    GCoptimizationGeneralGraph &gco) {
  for (size_t label = 0; label < group_centers.size(); ++label) {
    double cost = person.calculateDistanceCosts(group_centers[label], stride);
    for (size_t site = 0; site < other.size(); ++site) {
      cost += person.calculateVisibilityCost(group_centers[label], other[site]);
    }
    if (cost >= GCO_MAX_ENERGYTERM) {
      cost = (double)GCO_MAX_ENERGYTERM;
    }
    gco.setDataCost(person_id, label, cost);
  }
}

void setNeighbors(const std::vector<Person> &persons, const size_t person_id,
                  GCoptimizationGeneralGraph &gco) {
  for (size_t other_person = person_id + 1; other_person < persons.size();
       ++other_person) {
    gco.setNeighbors(person_id, other_person,
                     (persons[person_id].pose().position() -
                      persons[other_person].pose().position())
                         .norm());
  }
}

Classification createClassification(GCoptimizationGeneralGraph &gco,
                                    const std::vector<Person> &persons,
                                    const Timestamp &timestamp) {
  std::map<GCoptimization::LabelID, std::set<PersonId>> assignment;
  for (size_t person = 0; person < persons.size(); ++person) {
    auto assigned = gco.whatLabel(person);
    assignment[assigned].insert(persons[person].id());
  }
  std::vector<IdGroup> groups;
  for (auto group : assignment) {
    groups.push_back(IdGroup(group.second));
  }
  return Classification(timestamp, groups);
}

Classification GraphCutsOptimization::updateAssignment(
    const std::vector<Person> &persons,
    const std::vector<Position2D> &group_centers,
    const Timestamp &timestamp) const {
  // calculate assignment costs for all persons to all group centers
  GCoptimizationGeneralGraph gco(persons.size(), group_centers.size());
  for (size_t site = 0; site < persons.size(); ++site) {
    calculateCosts(persons[site], site, persons, group_centers, _stride, gco);
    setNeighbors(persons, site, gco);
  }
  gco.setLabelCost(_mdl);
  gco.expansion(10); // optimize
  return createClassification(gco, persons, timestamp);
}

template <typename T>
std::vector<T> shuffle(bool noop, std::vector<T> &src,
                       std::mt19937 &generator) {
  std::shuffle(src.begin(), src.end(), generator);
  return src;
}

size_t findSeed(const fformation::Options &options) {
  if (options.hasOption("seed")) {
    return options.getValue<size_t>("seed");
  } else {
    std::random_device rd;
    std::uniform_int_distribution<int> dist;
    return dist(rd);
  }
}

Classification
GraphCutsOptimization::detect(const Observation &observation) const {
  std::mt19937 rng(findSeed(options()));
  Classification classification = init(observation);
  if (observation.group().persons().size() < 2) {
    return classification;
  }
  double old_cost = std::numeric_limits<double>::max();
  double current_cost =
      classification.calculateCosts(observation, _stride, _mdl);
  std::vector<Person> persons;
  persons.reserve(observation.group().persons().size());
  for (auto person : observation.group().persons()) {
    persons.push_back(person.second);
  }
  shuffle(_shuffle, persons, rng);
  do {
    std::vector<Position2D> group_centers =
        updateGroupCenters(observation, classification, _stride);
    shuffle(_shuffle, group_centers, rng);
    classification =
        updateAssignment(persons, group_centers, observation.timestamp());
    old_cost = current_cost;
    current_cost = classification.calculateCosts(observation, _stride, _mdl);
  } while (current_cost < old_cost && classification.idGroups().size() > 1);
  return classification;
}
