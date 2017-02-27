# fformation-gco

This project uses [gco-v3.0](https://github.com/vrichter/gco-v3.0) to implement the classificator `gco` for [fformation](https://github.com/vrichter/fformation).

## How do I get set up? ###

    > git clone --recursive
    > mkdir -p fformation-gco/build && cd fformation-gco/build
    > cmake .. && make

## Applications

### fformation-gco-evaluation

```bash
Allowed options:
  -h [ --help ]                         produce help message
  -c [ --classificator ] arg (=list)    Which classificator should be used for
                                        evaluation. Possible:  ( gco | grow |
                                        none | one | )
  -e [ --evaluation ] arg (=threshold=0.6666)
                                        May be used to override evaluation
                                        settings and default settings from
                                        settings.json
  -d [ --dataset ] arg                  The root path of the evaluation
                                        dataset. The path is expected to
                                        contain features.json, groundtruth.json
                                        and settings.json
```

This application uses an evaluation dataset to evaluate a specific classificator
implementation. The dataset must be formatted as json and can be obtained from
[group-assignment-datasets](https://github.com/vrichter/group-assignment-datasets).

## Classificators

This project implements the following classificator:

#### gco

Implements the group assignment using a multi label graph-cuts optimization from
[1] as proposed in [2] and the corresponding matlab code
([GCFF](https://github.com/franzsetti/GCFF)).

## Citations

> [1] Delong A, Osokin A, Isack H. N., Boykov Y (2010) "Fast Approximate Energy Minimization with Label Costs". In CVPR.

<p/>

> [2] Setti F, Russell C, Bassetti C, Cristani M (2015) "F-Formation Detection:
Individuating Free-Standing Conversational Groups in Images". In PLoS ONE 10(5):
e0123783. [doi:10.1371/journal.pone.0123783](http://dx.doi.org/10.1371/journal.pone.0123783)

### 3rd party software used

* [Boost](http://www.boost.org/ "Boost C++ Libraries") because it is boost.
* [gco-v3.0](https://github.com/vrichter/gco-v3.0) for the graph-cuts optimization
* [fformation](https://github.com/vrichter/fformation) for the evaluation part

## Copyright

Please refer to [gco-v3.0](https://github.com/vrichter/gco-v3.0) if you want to
use this project.
