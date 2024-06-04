



### Dependencies :memo:

The main dependencies of the project are the following:

```yaml
python: 3.10.6
```

You can set up a conda environment as follows

```
conda create --name=proxyrecon python=3.9.12
conda activate proxyrecon
pip install -r requirements.txt
```

### Data :hammer:

Proxy reconstruction of input sfm sparse point cloud data in the data directory.

Download the newly proposed [building instance segmentation image dataset](https://drive.google.com/file/d/1TYDmgYtENNdJ3QCsB0jBf8HIh4dM9UGa/view?usp=drive_link) in the paper.

Download the newly proposed [virtual benchmark dataset](https://drive.google.com/file/d/1MVRsYej4agAUVLQKVYsU0jLtMc_WsDBH/view?usp=drive_link) in the paper.

#### Test :train2:

```sh
python experiment.py config/ct_1_inst_config.yaml
```

Save the output results in the experiment directory.

