This is the Caffe version used in the work "Robust Place Categorization with
Deep Domain Generalization".

This code allows to use Weighted Batch-Normalizaton layers. To add a weighted batch-norm use the following definition in the prototxt:


    layer{
        type: "MultiModalBatchNorm"
        bottom: "input"
        bottom: "weights"
        top: "output"
    }




With the second bottom "weights", being a vector of dimension Nx1, where N is the batch-size. Note that there must be 1 layer per per domain, with the weights already given as probabilities, and their output should be summed as in: 

    layer{
        type: "MultiModalBatchNorm"
        bottom: "input_1"
        bottom: "weights_1"
        top: "output_1"
    }

    layer{
        type: "MultiModalBatchNorm"
        bottom: "input_2"
        bottom: "weights_2"
        top: "output_2"
    }

    layer{
        type: "Eltwise"
        bottom: "output_1"
        bottom: "output_2"
        top: "output"
        eltwise_param{
            operation: SUM
        }
    }


An extended example can be found in core_modules.txt


# Caffe


Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
