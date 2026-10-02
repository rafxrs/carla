To change the diagram:

```bash
# MAC
pip install brew
brew install graphviz
dot -Tpng carla_uml.dot -o carla_uml.png
# Windows
conda install -c conda-forge graphviz
dot -V        # should print the Graphviz version
dot -Tpng carla_uml.dot -o carla_uml.png
```
