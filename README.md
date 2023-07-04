# Federated Learning with Intel&reg; Software Guard Extensions (SGX)

This repository contains the necessary code to train an end-to-end safe and robust Federated Learning (FL) pipeline based on the [**Intel&reg; Software Guard Extensions (SGX)**](https://www.intel.com/content/www/us/en/architecture-and-technology/software-guard-extensions.html), the library OS [Gramine](https://gramineproject.io/), the [OpenFL](https://openfl.readthedocs.io/en/latest/index.html) framework for FL, and [PyTorch](https://pytorch.org/). The workflow trains a [ResNet-18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) over four image classification datasets:
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [SVHN](http://ufldl.stanford.edu/housenumbers/)

For building a responsible and trustworthy FL infrastructure, the following points apply:
- **Memory** is encrypted with Intel&reg; SGX, a set of instructions on Intel&reg; processors for creating Trusted Execution Environments (TEE). In TEEs, the code executed and the data accessed are isolated and protected in terms of confidentiality (no one has access to the data except the code running inside the TEE) and integrity (no one can change the code and its behaviour).
- **Disk** is encrypted and transparently decrypted when accessed by Gramine or by any application running inside Gramine. Encrypted files guarantee data confidentiality and integrity (tamper resistance), as well as file swap protection (an encrypted file can only be accessed when in a specific host path).
- **Communication** among participants of the federation is achieved by Transport Layer Security (TLS), allowing for trusted transmission of arbitrary data.

The federation is composed by a server and three clients (respectively called _Aggregator_ and _Collaborators_ in OpenFL), running over four different physical servers (Bare Metal) (Intel&reg; Xeon(R) Platinum 8380 CPU @ 2.30GHz (Ice-Lake based), 2 thread(s) per core, 40 cores per socket, 2 sockets).

## Usage
- First of all, [find if your processor supports Intel&reg; SGX](https://www.intel.com/content/www/us/en/support/articles/000028173/processors.html).
- Clone this repo. Modify the environment variables of the `Dockerfile.gramine` and `openfl.manifest.template`, setting them equal to the number of core(s) per socket of your machine.
- [Install OpenFL](https://openfl.readthedocs.io/en/latest/install.html).
- Substitute the original `openfl/openfl-gramine/Dockerfile.gramine`, `openfl/openfl-gramine/Dockerfile.graminized.workspace` and `openfl/openfl-gramine/openfl.manifest.template` with the `Dockerfile.gramine`, `Dockerfile.graminized.workspace` and `openfl.manifest.template`provided in this repository. This step is necessary to avoid an undesired and heavy slowdown due to the creation/destruction of too many threads, as widely discussed [in this GitHub issue](https://github.com/gramineproject/gramine/issues/1253). <!---Moreover, another source of slowdown has been identified in Gramine, which chooses a slow path in resolving the system calls. In a typical centralized Deep Learning scenario, this problem can be solved by compiling Gramine with patched libgomp. However, this solution seems not working in a federated setting. To run experiments with Gramine built with patched libgomp, please use Dockerfile.gramine.libgomp and Dockerfile.graminized.workspace.libgomp (rename and remove .libgomp)-->
- Follow the [instructions](https://github.com/securefederatedai/openfl/blob/develop/openfl-gramine/MANUAL.md) to run OpenFL with Aggregator-based workflow inside SGX enclave with Gramine. To replicate our experiments, choose 3 as number of collaborators and `torch_cnn_mnist` as template. Before proceeding with point 2, follow the next step of this guide in order to implement the right dataset, network and plan. 
- In the building workspace, substitute `plan/plan.yaml`, `src/pt_cnn.py`, `src/NAME_OF_DATASET_utils.py` and `src/ptNAME_OF_DATASET_inmemory.py` with the same files provided in this repository. 
- In `plan.yaml` you can enable/disable TLS and in `openfl.manifest.template` you can enable/disable disk encryption. 

## Results
|   | MNIST | CIFAR10 | CIFAR100 | SVHN |
|---|-------|---------|----------|------|
| BASELINE | 1h55m41s | 2h1m5s | 1h54m42s | 3h7m24s |
| BASELINE (GRAMINE)  | 3h10m21s | 3h33m35s | 3h32m22s | 7h0m36s |
| SGX | 3h17m52s | 3h36m56s | 3h49m29s | 6h51m10s |
| SGX + FS | 3h21m12s | 3h41m59s | 3h40m34s | 7h39m49s |
| SGX + TLS| 3h42m54s | 3h53m52s | 3h59m55s | 8h10m53s |
| SGX + FS + TLS| 3h51m4s | 4h7m35s | 4h11m1s | 8h6m30s |

## Contributors
Bruno Casella <bruno.casella@unito.it>  

Iacopo Colonnelli <iacopo.colonnelli@unito.it> 

Gianluca Mittone <gianluca.mittone@unito.it> 

Robert Renè Maria Birke <robert.birke@unito.it> 

Marco Aldinucci <marco.aldinucci@unito.it>   