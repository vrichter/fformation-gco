// author: Viktor Richter vrichter@techfak.uni-bielefeld.de
// licence: MIT

#include "GraphCutsOptimization.h"
#include <boost/program_options.hpp>
#include <boost/timer.hpp>
#include <fformation/Features.h>
#include <fformation/GroundTruth.h>
#include <fformation/GroupDetectorFactory.h>
#include <fformation/JsonReader.h>
#include <fformation/Settings.h>
#include <future>
#include <iostream>

using namespace fformation;

GroupDetectorFactory &factory =
    GroupDetectorFactory::getDefaultInstance().addDetector(
        "gco", GraphCutsOptimization::creator());

struct DetectorConfig {
  std::string algorithm;
  double mdl;
  double stride;
  GroupDetector::Ptr detector;
  DetectorConfig(std::string _alg, double _mdl, double _stride)
      : algorithm(_alg), mdl(_mdl), stride(_stride),
        detector(factory.create(config_string(algorithm, mdl, stride))) {}

  static std::string config_string(std::string a, double m, double s) {
    std::stringstream conf;
    conf << a << "@mdl=" << m << "@stride=" << s;
    return conf.str();
  }
};

struct Frame {
  Timestamp time;
  Observation features;
  Classification ground_truth;
};

// returns the IdGroup of the person
Group create_persons_group(PersonId pid, const std::vector<Group> &groups) {
  for (auto g : groups) {
    if (g.persons().find(pid) != g.persons().end()) {
      return g;
    }
  }
  return Group();
}

// creates a confusion matrix for one pid and its group participants
ConfusionMatrix
calculate_persons_cm(const PersonId &pid,
                     const std::vector<Person> &persons_in_frame,
                     const Group &cl, const Group &an) {
  ConfusionMatrix::IntType tp = 0;
  ConfusionMatrix::IntType fp = 0;
  ConfusionMatrix::IntType tn = 0;
  ConfusionMatrix::IntType fn = 0;
  for (auto person : persons_in_frame) {
    // if (person.id() != pid) {
    bool in_cl = cl.persons().find(person.id()) != cl.persons().end();
    bool in_an = an.persons().find(person.id()) != an.persons().end();
    tp += (in_cl & in_an) ? 1 : 0;
    fp += (in_cl & !in_an) ? 1 : 0;
    tn += (!in_cl & !in_an) ? 1 : 0;
    fn += (!in_cl & in_an) ? 1 : 0;
    //}
  }
  return ConfusionMatrix(tp, fp, tn, fn);
}

// create groups while ignoring persons missing from observation
std::vector<Group> generate_group_lists(const Classification &cl,
                                        const Observation &ob) {
  std::vector<Group> result;
  for (auto idg : cl.idGroups()) {
    std::vector<Person> persons;
    for (auto pid : idg.persons()) {
      auto it = ob.group().persons().find(pid);
      if (it != ob.group().persons().end()) {
        persons.push_back(it->second);
      }
    }
    if (!persons.empty()) {
      result.push_back(Group(persons));
    }
  }
  for (auto it : ob.group().persons()) {
    auto pid = it.first;
    auto person = it.second;
    bool found = false;
    for (auto g : result) {
      if (g.has_person(pid)) {
        found = true;
        break;
      }
    }
    if (!found) {
      result.push_back(Group(std::vector<Person>({person})));
    }
  }
  return result;
}

// counts the persons that are in ground truth but not in observation
size_t count_missing_persons(const PersonId &pid, const Classification &gt,
                             const Observation &ob) {
  size_t result = 0;
  for (auto idg : gt.idGroups()) {
    if (idg.has_person(pid)) {
      for (auto group_participant_id : idg.persons()) {
        if (!ob.group().has_person(group_participant_id)) {
          ++result;
        }
      }
    }
  }
  return result;
}

struct Evaluation {
  struct Line {
    size_t group_size;
    size_t detected_group_size;
    size_t missing_members;
  };

  std::map<PersonId, Line> results;
  Evaluation(const Classification &gt, const Classification &cl,
             const Observation &obs, const std::set<PersonId> &ids) {
    const auto person_list = obs.group().generatePersonList();
    const auto gt_groups = generate_group_lists(gt, obs);
    const auto cl_groups = generate_group_lists(cl, obs);
    for (auto pid : ids) {
      const Group &gt_group = create_persons_group(pid, gt_groups);
      const size_t missing = count_missing_persons(pid, gt, obs);
      const Group &cl_group = create_persons_group(pid, cl_groups);
      Line l;
      l.group_size = gt_group.persons().size();
      l.detected_group_size = cl_group.persons().size();
      l.missing_members = missing;
      results[pid] = l;
    }
  }
};

struct FrameResult {
  Frame frame;
  Classification classification;

  FrameResult(GroupDetector &d, const Frame &f)
      : frame(f), classification(d.detect(f.features)) {}
};

struct Result {
  std::string alg;
  double mdl;
  double stride;
  std::string frame_type;
  std::vector<FrameResult> frames;

  std::string config() {
    std::stringstream str;
    str << alg << " - " << mdl << " - " << stride << " - " << frame_type;
    return str.str();
  }
};

class DataSet {
  using Frames = std::vector<Frame>;
  std::map<std::string, Frames> _frames;

public:
  DataSet(std::string &af, std::string &df, std::string &gt) {
    boost::timer timer;
    timer.restart();
    Features annotated_features = Features::readMatlabJson(af);
    std::cerr << "  --read features 1" << timer.elapsed() << std::endl;
    timer.restart();
    Features detected_features = Features::readMatlabJson(df);
    std::cerr << "  --read features 2" << timer.elapsed() << std::endl;
    timer.restart();
    GroundTruth groundtruth = GroundTruth::readMatlabJson(gt);
    std::cerr << "  --read ground truth " << timer.elapsed() << std::endl;
    timer.restart();
    std::map<Timestamp, Observation> annotation_map;
    for (auto d : annotated_features.observations()) {
      annotation_map[d.timestamp()] = d;
    }
    std::cerr << "  --create annotation map " << timer.elapsed() << std::endl;
    timer.restart();
    std::map<Timestamp, Observation> detection_map;
    for (auto d : detected_features.observations()) {
      detection_map[d.timestamp()] = d;
    }
    std::cerr << "  --create detection map " << timer.elapsed() << std::endl;
    timer.restart();
    Frames &frames_annotated = _frames["annotated"];
    Frames &frames_detected = _frames["detected"];
    frames_annotated.reserve(groundtruth.classifications().size());
    frames_detected.reserve(groundtruth.classifications().size());
    for (auto cl : groundtruth.classifications()) {
      auto t = cl.timestamp();
      auto a_it = annotation_map.find(t);
      auto d_it = detection_map.find(t);
      if (a_it == annotation_map.end()) {
        std::cerr << "Annotation features do not contain timestamp " << t
                  << std::endl;
        continue;
      }
      if (d_it == detection_map.end()) {
        std::cerr << "Detection features do not contain timestamp " << t
                  << std::endl;
        continue;
      }
      frames_annotated.push_back({t, d_it->second, cl});
      frames_detected.push_back({t, d_it->second, cl});
    }
    std::cerr << "  --create frames " << timer.elapsed() << std::endl;
    timer.restart();
    std::sort(frames_annotated.begin(), frames_annotated.end(),
              [](Frame a, Frame b) -> bool { return a.time < b.time; });
    std::sort(frames_detected.begin(), frames_detected.end(),
              [](Frame a, Frame b) -> bool { return a.time < b.time; });
    std::cerr << "  --sort frames " << timer.elapsed() << std::endl;
  }

  const std::map<std::string, std::vector<Frame>> &frames() { return _frames; };
};

template <typename T>
std::vector<T> read_array(const Json &js, std::string elem) {
  std::vector<T> result;
  auto it = js.find(elem);
  if (it != js.end()) {
    Exception::check(it.value().is_array(),
                     elem + " must be an array. Got: " + js.dump());
    result.reserve(it.value().size());
    for (auto ts : it.value()) {
      result.push_back(ts);
    }
  }
  return result;
}

std::vector<DetectorConfig> setup_detectors(Json &settings) {
  auto mdls = read_array<double>(settings, "mdls");
  if (mdls.empty()) {
    mdls.push_back(settings["mdl"]);
  }
  auto strides = read_array<double>(settings, "strides");
  if (strides.empty()) {
    strides.push_back(settings["stride"]);
  }
  auto algorithms = read_array<std::string>(settings, "algorithms");
  if (algorithms.empty()) {
    algorithms.push_back("gco");
  }
  std::vector<DetectorConfig> configs;
  for (auto a : algorithms) {
    for (auto m : mdls) {
      for (auto s : strides) {
        configs.push_back(DetectorConfig(a, m, s));
      }
    }
  }
  return configs;
}

struct Statistics {
  ConfusionMatrix in_group;
};

ConfusionMatrix create_cm(bool condition, bool prediction) {
  if (prediction) {
    if (condition) {
      return ConfusionMatrix(1, 0, 0, 0);
    } else {
      return ConfusionMatrix(0, 1, 0, 0);
    }
  } else {
    if (condition) {
      return ConfusionMatrix(0, 0, 0, 1);
    } else {
      return ConfusionMatrix(0, 0, 1, 0);
    }
  }
}

std::map<PersonId, Statistics>
calculate_statistics(const std::vector<FrameResult> &results,
                     const std::set<PersonId> &ids) {
  std::map<PersonId, Statistics> result;
  for (auto frame_result : results) {
    auto &obs = frame_result.frame.features;
    auto &gt = frame_result.frame.ground_truth;
    auto &cl = frame_result.classification;
    const auto person_list = obs.group().generatePersonList();
    const auto gt_groups = generate_group_lists(gt, obs);
    const auto cl_groups = generate_group_lists(cl, obs);
    auto local_ids = ids;
    if (ids.empty()) {
      for (auto p_it : obs.group().persons()) {
        local_ids.insert(p_it.first);
      }
    }
    for (auto id : local_ids) {
      auto &stat = result[id];
      auto gt_group_person = create_persons_group(id, gt_groups);
      auto cl_group_person = create_persons_group(id, cl_groups);
      stat.in_group =
          stat.in_group + create_cm(gt_group_person.persons().size() > 1,
                                    cl_group_person.persons().size() > 1);
    }
  }
  return result;
}

template <typename T> void print(const T t) { std::cout << t << "\t"; }

void print_results(const Result &r, const std::set<PersonId> &ids) {
  auto stats = calculate_statistics(r.frames, ids);
  for (auto stat_it : stats) {
    auto &stat = stat_it.second;
    print(r.alg);
    print(r.mdl);
    print(r.stride);
    print(r.frame_type);
    print(stat_it.first);
    print(stat.in_group.calculatePrecision());
    print(stat.in_group.calculateRecall());
    print(stat.in_group.calculateF1Score());
    std::cout << std::endl;
  }
}

std::vector<FrameResult> eval_batch(GroupDetector *detector, Result *result,
                                    const std::vector<Frame> *frames,
                                    size_t from, size_t to) {
  std::cerr << "    starting batch " << result->config() << " " << from << " - "
            << to << std::endl;
  std::vector<FrameResult> results;
  results.reserve(to - from);
  for (size_t i = from; i < to; ++i) {
    results.push_back(FrameResult(*detector, frames->at(i)));
  }
  std::cerr << "    done batch " << result->config() << " " << from << " - "
            << to << std::endl;
  return results;
}

Result eval_config(DetectorConfig *detector, DataSet *dataset,
                   std::string frame_type, std::launch async_batches,
                   size_t batch_size) {
  Result result;
  result.alg = detector->algorithm;
  result.mdl = detector->mdl;
  result.stride = detector->stride;
  result.frame_type = frame_type;
  std::cerr << "  start calculating " << result.config() << std::endl;
  const std::vector<Frame> &frames = dataset->frames().at(frame_type);
  result.frames.reserve(frames.size());
  std::vector<std::future<std::vector<FrameResult>>> batches;
  for (size_t from = 0; from < frames.size(); from = from + batch_size) {
    size_t to = std::min(from + batch_size, frames.size() - 1);
    batches.push_back(std::async(async_batches, eval_batch,
                                 detector->detector.get(), &result, &frames,
                                 from, to));
  }
  for (auto &future_result : batches) {
    for (auto frame_result : future_result.get()) {
      result.frames.push_back(frame_result);
    }
  }
  std::cerr << "  done calculating " << result.config() << std::endl;
  return result;
}

int main(const int argc, const char **args) {
  boost::program_options::variables_map program_options;
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message");
  desc.add_options()("async-configs",
                     "Calculate configuration sets in threads.");
  desc.add_options()("async-batches", "Calculate frame batches in threads.");
  desc.add_options()(
      "batch-size",
      boost::program_options::value<size_t>()->default_value(10000),
      "Calculate frame batches in threads.");
  desc.add_options()(
      "dataset,d", boost::program_options::value<std::string>()->required(),
      "The root path of the evaluation dataset. The path is expected "
      "to contain features.json, features2.json, groundtruth.json, and "
      "settings.json");
  try {
    boost::program_options::store(
        boost::program_options::parse_command_line(argc, args, desc),
        program_options);
    boost::program_options::notify(program_options);
  } catch (const std::exception &e) {
    std::cerr << "Error while parsing command line parameters:\n\t" << e.what()
              << "\n";
    std::cerr << desc << std::endl;
    return 1;
  }

  if (program_options.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }
  auto async_configs = (program_options.count("async-configs"))
                           ? std::launch::async
                           : std::launch::deferred;
  auto async_batches = (program_options.count("async-batches"))
                           ? std::launch::async
                           : std::launch::deferred;
  auto batch_size = program_options["batch-size"].as<size_t>();
  Exception::check(batch_size > 0, "--batch-size cannot be <= 0");

  std::string path = program_options["dataset"].as<std::string>();
  std::string features_path1 = path + "/features.json";
  std::string features_path2 = path + "/features2.json";
  std::string groundtruth_path = path + "/groundtruth.json";
  std::string settings_path = path + "/settings.json";
  boost::timer timer;
  Json settings = JsonReader::readFile(settings_path);
  std::cerr << "--read settings " << timer.elapsed() << std::endl;
  timer.restart();

  std::set<PersonId> ids;
  for (auto id : read_array<std::string>(settings, "persons")) {
    ids.insert(PersonId(id));
  }
  std::cerr << "--read persons " << timer.elapsed() << std::endl;
  timer.restart();

  auto detectors = setup_detectors(settings);
  std::cerr << "--setup detectors #" << detectors.size() << " detectors in "
            << timer.elapsed() << std::endl;
  timer.restart();

  DataSet data(features_path1, features_path2, groundtruth_path);
  std::cerr << "--read datasets " << timer.elapsed() << std::endl;
  timer.restart();

  std::cerr << "--creating evaluation " << std::endl;
  std::vector<std::future<Result>> future_results;
  for (auto frame_it : data.frames()) {
    for (auto &detector : detectors) {
      future_results.push_back(std::async(async_configs, eval_config, &detector,
                                          &data, frame_it.first, async_batches,
                                          batch_size));
    }
  }
  std::cerr << "--evaluation created 2*" << future_results.size()
            << " futures in " << timer.elapsed() << std::endl;
  timer.restart();
  for (auto &fr : future_results) {
    print_results(fr.get(), ids);
  }
  std::cerr << "--evaluation completed " << timer.elapsed() << std::endl;
  timer.restart();
}
