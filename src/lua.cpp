#include "GraphCutsOptimization.h"
#include "luwra.hpp"
#include <iostream>

using State = luwra::State;
using Table = luwra::Table;
using fformation::GroupDetectorFactory;
using Observation = fformation::Observation;
using Classification = fformation::Classification;
using fformation::Timestamp;
using fformation::PersonId;
using fformation::Position2D;
using fformation::OptionalRotationRadian;
using fformation::RotationRadian;
using fformation::Person;
using fformation::Group;

OptionalRotationRadian get_optional_rotation(Table &t) {
  if (t.has("rad")) {
    return OptionalRotationRadian(true, t.get<RotationRadian>("rad"));
  } else {
    return OptionalRotationRadian();
  }
}

Person table_to_person(Table table) {
  auto id = PersonId(table.get<PersonId::PersonIdType>("id"));
  auto x = table.get<Position2D::Coordinate>("x");
  auto y = table.get<Position2D::Coordinate>("y");
  auto r = get_optional_rotation(table);
  return Person(id, {{x, y}, r});
}

Group table_to_group(Table table) {
  std::vector<Person> persons;
  for (size_t i = 1;; ++i) {
    if (table.has(i)) {
      persons.push_back(table_to_person(table.get<Table>(i)));
    } else {
      break;
    }
  }
  return Group(persons);
}

Observation table_to_observation(Table &table) {
  Timestamp time = 0;
  if (table.has("time")) {
    time = Timestamp(table.get<Timestamp::TimestampType>("time"));
  }
  if (table.has("persons")) {
    return Observation(time, table_to_group(table.get<Table>("persons")));
  } else {
    return Observation(time);
  }
}

Table classification_to_table(Classification c, State *s) {
  Table result = Table(s);
  for (size_t i = 0; i < c.idGroups().size(); ++i) {
    Table group = Table(s);
    for (auto pid : c.idGroups().at(i).persons()) {
      group[pid.as<std::string>()] = true;
    }
    result[i + 1] = group;
  }
  return result;
}

struct GroupDetector {
  fformation::GroupDetector::Ptr _detector;
  GroupDetector(std::string config) {
    auto &factory = GroupDetectorFactory::getDefaultInstance().addDetector(
        "gco", GraphCutsOptimization::creator());
    _detector = factory.create(config);
  }

  Table detect(Table observation) {
    return classification_to_table(
        _detector->detect(table_to_observation(observation)),
        observation.ref.life->state);
  }

  Table list_alg(Table t) {
    auto det = GroupDetectorFactory::getDefaultInstance().listDetectors();
    for (size_t i = 0; i < det.size(); ++i) {
      t[i + 1] = det[i];
    }
    return t;
  }
};

extern "C" {
int luaopen_gco_lua(lua_State *lua) {
  luwra::registerUserType<GroupDetector(std::string)>(
      lua, "GroupDetector", {LUWRA_MEMBER(GroupDetector, detect),
                             LUWRA_MEMBER(GroupDetector, list_alg)});
  return 0;
}
}
