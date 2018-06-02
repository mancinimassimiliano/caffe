This is the Caffe version used in the work [Robust Place Categorization with Deep Domain Generalization](https://ieeexplore.ieee.org/document/8302933/). 

This code is forked from [NVIDIA/caffe](https://github.com/NVIDIA/caffe). For any issue not directly related to our additional layer (MultiModalBatchNorm), please refer to the upstream repository.

This code allows to use Weighted Batch-Normalizaton layers. To add a weighted batch-norm use the following definition in the prototxt:


    layer{
        name: "wbn"
        type: "MultiModalBatchNorm"
        bottom: "input"
        bottom: "weights"
        top: "output"
    }




With the second bottom "weights", being a vector of dimension Nx1, where N is the batch-size. Note that there must be 1 layer per domain, with the weights already given as probabilities, and their output should be summed as in: 

    layer{
        name: "wbn1"
        type: "MultiModalBatchNorm"
        bottom: "input_1"
        bottom: "weights_1"
        top: "output_1"
    }

    layer{
        name: "wbn2"
        type: "MultiModalBatchNorm"
        bottom: "input_2"
        bottom: "weights_2"
        top: "output_2"
    }

    layer{
        name: "wbn"
        type: "Eltwise"
        bottom: "output_1"
        bottom: "output_2"
        top: "output"
        eltwise_param{
            operation: SUM
        }
    }


An extended example can be found in core_modules.txt.


## Abstract and citation

Traditional place categorization approaches in robot vision assume that training and test images have similar visual appearance. Therefore, any seasonal, illumination, and environmental changes typically lead to severe degradation in performance. To cope with this problem, recent works have been proposed to adopt domain adaptation techniques. While effective, these methods assume that some prior information about the scenario where the robot will operate is available at training time. Unfortunately, in many cases, this assumption does not hold, as we often do not know where a robot will be deployed. To overcome this issue, in this paper, we present an approach that aims at learning classification models able to generalize to unseen scenarios. Specifically, we propose a novel deep learning framework for domain generalization. Our method develops from the intuition that, given a set of different classification models associated to known domains (e.g., corresponding to multiple environments, robots), the best model for a new sample in the novel domain can be computed directly at test time by optimally combining the known models. To implement our idea, we exploit recent advances in deep domain adaptation and design a convolutional neural network architecture with novel layers performing a weighted version of batch normalization. Our experiments, conducted on three common datasets for robot place categorization, confirm the validity of our contribution.

    @article{mancini2018robust,
      title={Robust Place Categorization With Deep Domain Generalization},
      author={Mancini, Massimiliano and Rota Bul{\`o}, Samuel and Caputo, Barbara and Ricci, Elisa},
      journal={IEEE Robotics and Automation Letters},
      year={2018},
      volume={3},
      number={3},
      pages={2093-2100},
      doi={10.1109/LRA.2018.2809700},
      month={July},
    }


