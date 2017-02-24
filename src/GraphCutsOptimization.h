// author: Viktor Richter vrichter@techfak.uni-bielefeld.de
// licence: MIT

#include <fformation/GroupDetector.h>
#include <fformation/GroupDetectorFactory.h>

class GraphCutsOptimization : public fformation::GroupDetector {
public:
  GraphCutsOptimization(const fformation::Options optinos);

  fformation::GroupDetectorFactory::ConstructorFunction static creator();

  virtual fformation::Classification
  detect(const fformation::Observation &observation) const final;

  fformation::Classification
  init(const fformation::Observation observation) const;

  fformation::Classification
  updateAssignment(const std::vector<fformation::Person> &persons,
                   const std::vector<fformation::Position2D> &group_centers,
                   const fformation::Timestamp &timestamp) const;
  std::vector<fformation::Position2D>
  updateGroupCenters(const fformation::Observation &observation,
                     const fformation::Classification &assignment,
                     const fformation::Person::Stride &stride) const;

private:
  double _mdl;
  fformation::Person::Stride _stride;
  bool _shuffle;
};
