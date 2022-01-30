![example workflow](https://github.com/jnyfah/Enarx-Demo/actions/workflows/rust.yml/badge.svg)
# Enarx-Demo 

This demo shows how to easily deploy a simple machine learning workload to Trusted Execution Enviroment on a host using Enarx. In this example it is assumed that the user wants to protect the machine model by training the model in a TEE using an external data the workload has been given access to.

The Tee is where the training and processing takes place, this ensures model remains confidential during processing as this gurantees data confidentiality, code integrity and data integrity.

In this demo a simple ML model using decision tree algorithm was created on the [PIMA Indian Diabetes dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database) to predict if a particular observation is at a risk of developing diabetes, given the independent factors.

The decision tree Algorithm belongs to the family of supervised machine learning algorithms. It can be used for both a classification problems as well as for regression problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome. You can read more on decision algorithm trees [here](https://towardsdatascience.com/a-guide-to-decision-trees-for-machine-learning-and-data-science-fe2607241956)

## Environment setup

To compile this demo, you must install the following.

#### Rust

Go to [rust-lang.org](https://www.rust-lang.org/tools/install) and follow the instructions using rustup.

#### WebAssembly System Interface (WASI)

install Wastime

```bash
curl https://wasmtime.dev/install.sh -sSf | bash
```

#### Install Enarx
 
 Visit [Enarx Project home page](https://enarx.dev/docs/Installation/Enarx) and follow the instructons

#### Code

```bash
 git clone https://github.com/jnyfah/Enarx-Demo
 cd Enarx-Demo
 cargo build
 cargo run data/diabetes.csv
```

*Please see [this wiki page](https://github.com/jnyfah/Enarx-Demo/wiki) for instructions.*


*Please see [this wiki page](https://github.com/jnyfah/Enarx-Demo/wiki) for instructions.*

