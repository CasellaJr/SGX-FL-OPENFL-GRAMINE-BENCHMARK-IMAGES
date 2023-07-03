# SGX-FL-OPENFL-GRAMINE-BENCHMARK-IMAGES

This repository contains the necessary code to train an end-to-end safe and robust Federated Learning (FL) pipeline based on the [**Intel&reg; Software Guard Extensions (SGX)**](https://www.intel.com/content/www/us/en/architecture-and-technology/software-guard-extensions.html), the library OS [Gramine](https://gramineproject.io/), the [OpenFL](https://openfl.readthedocs.io/en/latest/index.html) framework for FL, and [PyTorch](https://pytorch.org/). The workflow trains a [ResNet-18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) over four image classification datasets:
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [SVHN](http://ufldl.stanford.edu/housenumbers/)

For building a responsible and trustworthy FL infrastructure, the following points apply:
- **Memory** is encrypted with Intel&reg; SGX, a set of instructions on Intel&reg; processors for creating Trusted Execution Environments (TEE). In TEEs, the code executed and the data accessed are isolated and protected in terms of confidentiality (no one has access to the data except the code running inside the TEE) and integrity (no one can change the code and its behaviour).
- **Disk** is encrypted and transparently decrypted when accessed by Gramine or by any application running inside Gramine. Encrypted files guarantee data confidentiality and integrity (tamper resistance), as well as file swap protection (an encrypted file can only be accessed when in a specific host path).
- **Communication** among participants of the federation is achieved by Transport Layer Security (TLS), allowing for trusted transmission of arbitrary data.

The federation is composed by a server and three clients (respectively called _Aggregator_ and _Collaborators_ in OpenFL), running over four different physical servers (Bare Metal) (Intel&reg; Xeon(R) Platinum 8380 CPU @ 2.30GHz (Ice-Lake based), 2 thread(s) per core, 40 cores per socket, 2 sockets)