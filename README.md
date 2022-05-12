**Open-Data Cost Model**

This repository contains the Open-Data Cost Model for Wireless Backhaul Networks associated with the homonym article submitted to the 20th International Symposium on Modeling and Optimization in Mobile, Ad hoc, and Wireless Networks.

The repository is organized as follows:

- `data/` : contains the visibility graphs representing the areas of analysis together with the node attributes regarding the household and population distribution
- `Sim.py `: The main file for the generations of the WBN topologies
- `Topology.py` : the logic for the generation of the topology
- `sim.yaml.example` : the file containing the experimental variables
- `results_110522/` : the folder containing the topologies generated for the WiOPT paper
- `analysis/` : the folder containing the Jupyter notebook implementing the Cost Model and the analysis of the topologies.

**How to replicate the results**
The WBN topologies generated for our research paper are available in the folder `results_110522`. You can either analyze those by copying them into the `results` folder or generate new ones by following the instruction in the next section.

To analyze the topologies you can execute the Jupyter notebook : `analysis/analysis_fiber.ipynb`

**How to generate the WBN topologies**

In order to generate the topologies, first you need to install the python dependencies:

```
pip -r requirements.txt
```

Then, copy the `sim.yaml.example` file to `sim.yaml`, and adjust the content to your needs.

```
cp sim.yaml.exampe sim.yaml
```

Run the `batch.sh` file that will generate the topology for the parameters new set provided in sim.yaml:

```
./batch.sh
```

This will generate a set of topologies in the `results/` folder.
