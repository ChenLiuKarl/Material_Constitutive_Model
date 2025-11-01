Here are two neural networks for training material constitutive model based on measured strain and stress.

For Dense, it uses a Fully-Connected Neural Network (FCNN). Usually the material has 6 dimensions of strain and stress (input and output size). The hyper parameters that can be adjusted are: number of layers, layer size, non linearity and learn rate.
Currently the model uses SGD optimizer and LR scheduler.

There is another Recurrent Neural Network (RNN) for training constitutive model for materials with memory (e.g. viscoelastic material). In RNN, the hyper parameters are: hidden variable number, hidden network architecture, input network architecture and learn rate.
Currently the model uses Adam optimizer and LR scheduler.

The author wants to thank Dr.Burigede Liu and Miss Rui Wu for providing the code skeleton in Course 4C11 of CUED.
