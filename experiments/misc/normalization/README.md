# Experiments for Mini-Batch Normalzation Schemes

> NOTE 1: The discussions below are for semantic segmentation task for predicting binary foreground and instance segmentation task for predicting foreground and boundary.

> NOTE 2: The chosen evaluation metric is `Dice Score (DSC)` for semantic segmentation,
and `Segmentation Accuracy` for instance segmentation.

## Quantitative Results:

### LIVECell:

| Targets  | Mini-Batch Norm. | Input Norm.     | Dice               | mSA                 | SA50                |
|----------|------------------|-----------------|--------------------|---------------------|---------------------|
| Binary   | OldDefault       | Per Tile        | 0.928867041453946  | -                   | -                   |
| Binary   | InstanceNorm     | Per Tile        | 0.9139200411471561 | -                   | -                   |
| Boundary | OldDefault       | Per Tile        | -                  | 0.3596574029856674  | 0.5706313962830515  |
| Boundary | InstanceNorm     | Per Tile        | -                  | 0.21136391380220415 | 0.32975049087380925 |
| Binary   | OldDefault       | Whole Volume    | 0.9288671102955196 | -                   | -                   |
| Binary   | InstanceNorm     | Whole Volume    | 0.9139609056661331 | -                   | -                   |
| Boundary | OldDefault       | Whole Volume    | -                  | 0.3596633257789534  | 0.5706479532532667  |
| Boundary | InstanceNorm     | Whole Volume    | -                  | 0.21294970907224356 | 0.33214134838916115 |



### Mouse Embryo (Nuclei):

| Targets  | Mini-Batch Norm. | Input Norm.     | Dice               | mSA                 | SA50                |
|----------|------------------|-----------------|--------------------|---------------------|---------------------|
| Binary   | OldDefault       | Per Tile        | 0.773449732035369  | -                   | -                   |
| Binary   | InstanceNorm     | Per Tile        | 0.7413715897174049 | -                   | -                   |
| Boundary | OldDefault       | Per Tile        | -                  | 0.2089494345163684  | 0.396692895729518   |
| Boundary | InstanceNorm     | Per Tile        | -                  | 0.15872795687689387 | 0.3200512111903994  |
| Binary   | OldDefault       | Whole Volume    | 0.7734503020445563 | -                   | -                   |
| Binary   | InstanceNorm     | Whole Volume    | 0.767180298383585  | -                   | -                   |
| Boundary | OldDefault       | Whole Volume    | -                  | 0.20882369840046283 | 0.39638644753570157 |
| Boundary | InstanceNorm     | Whole Volume    | -                  | 0.15714229749213487 | 0.3231149929195066  |


## PlantSeg (Root):

| Targets  | Mini-Batch Norm. | Input Norm.     | Dice               | mSA                   | SA50                  |
|----------|------------------|-----------------|--------------------|-----------------------|-----------------------|
| Binary   | OldDefault       | Per Tile        | 0.9928809913563437 | -                     | -                     |
| Binary   | InstanceNorm     | Per Tile        | 0.981061366019915  | -                     | -                     |
| Boundary | OldDefault       | Per Tile        | -                  | 0.014372028994123383  | 0.018482085449422186  |
| Boundary | InstanceNorm     | Per Tile        | -                  | 0.0014261363636363636 | 0.004753787878787879  |
| Binary   | OldDefault       | Whole Volume    | 0.9928820421405989 | -                     | -                     |
| Binary   | InstanceNorm     | Whole Volume    | 0.9810579911503293 | -                     | -                     |
| Boundary | OldDefault       | Whole Volume    | -                  | 0.014372028994123383  | 0.018482085449422186  |
| Boundary | InstanceNorm     | Whole Volume    | -                  | 0.0014396498771498771 | 0.0047988329238329245 |


## MitoEM (Human and Rat):

| Targets  | Mini-Batch Norm. | Input Norm.     | Dice               | mSA                 | SA50                |
|----------|------------------|-----------------|--------------------|---------------------|---------------------|
| Binary   | OldDefault       | Per Tile        | 0.9206916296864652 | -                   | -                   |
| Binary   | InstanceNorm     | Per Tile        | 0.9247380570327044 | -                   | -                   |
| Boundary | OldDefault       | Per Tile        | -                  | 0.4452180959378787  | 0.5943295393094057  |
| Boundary | InstanceNorm     | Per Tile        | -                  | 0.23307025211315505 | 0.32876735818930347 |
| Binary   | OldDefault       | Whole Volume    | 0.9206915268071336 | -                   | -                   |
| Binary   | InstanceNorm     | Whole Volume    | 0.9241492985508706 | -                   | -                   |
| Boundary | OldDefault       | Whole Volume    | -                  | 0.44530557897744855 | 0.594423580706029   |
| Boundary | InstanceNorm     | Whole Volume    | -                  | 0.24493949819392608 | 0.3480087025599067  |
