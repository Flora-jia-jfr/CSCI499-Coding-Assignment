# Model Report

Furong(Flora) Jia | CSCI 499 Coding Assignment 1

## Implementation Detail

### Model architecture


### Performance of model based on selection of hyperparameters
I rented a server with GPU to train the models, and I have stored the corresponding models in
the `/experiements` folder.  
The following are the parameter combinations that I have tried and their performance accordingly.  
1. lr = 0.001  
   batch_size = 1024  
   num_epochs = 50  
   val_every = 5  
   embedding_dim = 100  
   num_hiddens = 128  
   num_layers = 2  
   ![image](result_plots/train&valid_lost&accuracy_1.png)  
   the model is stored as `/experiments/lstmentire_model_1.pt`.  
   The final test action accuracy is around 0.98, and the final test target accuracy is around 0.92.  
   The final valid action accuracy is around 0.96, and the final valid target accuracy is around 0.76.  
   The validation loss of both target and action begins to increase after around epoch 10,
   but the train loss continues to decrease. Similarly, the validation accuracy of 
   action begins to decrease after around 10 epochs while the train accuracy continues
   to increase. While there is not much difference for the valid action accuracy, the train
   action accuracy growths through epochs. All of this shows that the model is probably
   encountering over-fitting after 10 epochs.  
   Therefore, I will apply two different changes on my parameters respectively. I will
   first try changing it into a 10-epoch model, and see the accuracy. Another choice would
   be to change the learning rate, to see if smaller learning rate can relieve the over-fitting
   problem and keep improving the model.
2. lr = 0.001  
   batch_size = 1024  
   num_epochs = 10  
   val_every = 2  
   embedding_dim = 100  
   num_hiddens = 128  
   num_layers = 2  
   ![image](result_plots/train&valid_lost&accuracy_2.png)  
   The model is stored as `/experiments/lstmentire_model_2.pt`.  
   The final test action accuracy is around 0.98, and the final test target accuracy is around 0.83.  
   The final valid action accuracy is around 0.96, and the final valid target accuracy is around 0.76.  
   This time, all the loss and accuracy begins to behave smoothly, which indicates that
   there is not much improvement in the model. The model still didn't perform as well on the valid
   dataset as on the test dataset. One possible reason is that the there are certain unseen words in
   the valid dataset that weakens the model's performance.  
3. lr = 0.0001 (very small learning rate)  
   batch_size = 1024  
   num_epochs = 50  
   val_every = 2 (a more detailed validation)  
   embedding_dim = 100  
   num_hiddens = 128  
   num_layers = 2  
   ![image](result_plots/train&valid_lost&accuracy_3.png)  
   The model is stored as `/experiments/lstmentire_model_3.pt`.  
   The final test action accuracy is around 0.98, and the final test target accuracy is around 0.82.  
   The final valid action accuracy is around 0.96, and the final valid target accuracy is around 0.76.  
   This time, the model didn't show a decrease in accuracy on the valid dataset. It still begins to 
   behave smoothly after roughly 10 or 20 epochs.
4. lr = 0.0001  
   batch_size = 1024  
   num_epochs = 50  
   val_every = 2  
   embedding_dim = 100  
   num_hiddens = 256  
   num_layers = 3  
   I try to add to the model's complexity to see if this can improve the model's performance.  
    ![image](result_plots/train&valid_lost&accuracy_4.png)  
   The model is stored as `/experiments/lstmentire_model_4.pt`.   
   The final test action accuracy is around 0.98, and the final test target accuracy is around 0.89.  
   The final valid action accuracy is around 0.96, and the final valid target accuracy is around 0.73.  
   This shows that when increase the complexity of the model by adding more layers and more 
   hidden units, the performance didn't go up. Therefore, it is not necessary to use a such complicated
   model for this classification task.

   
