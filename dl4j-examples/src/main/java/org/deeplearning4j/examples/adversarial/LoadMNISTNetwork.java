package org.deeplearning4j.examples.adversarial;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.ObjectInputStream;


/**
 * Created by Kamal Kamalaldin on 10/30/2016.
 * The network testing code is taken from the convolution/LenetMnistExample.java.
 */
public class LoadMNISTNetwork {
    private static final Logger log = LoggerFactory.getLogger(TrainAndSaveMNIST.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int outputNum = 10;
        int nEpochs =5;

        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);

        // Read from disk using FileInputStream
        //make sure you run TrainAndSaveMNIST.java
        FileInputStream f_in = new
            FileInputStream("network.data");


        // Read object using ObjectInputStream
        ObjectInputStream obj_in =
            new ObjectInputStream (f_in);

        // Read an object
        Object obj = obj_in.readObject();

        MultiLayerNetwork model = null;
        if (obj instanceof MultiLayerNetwork)
        {
            // Cast object to a a neural network object
            model = (MultiLayerNetwork) obj;
            System.out.println("Retrieval successful");
        }
        else
        {
            System.out.println("The file provided does not contain an object of type"+
                "MultiLayerNetwork");
            return;
        }


        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(mnistTrain);
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while(mnistTest.hasNext()){
                DataSet ds = mnistTest.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
            }
            log.info(eval.stats());
            mnistTest.reset();
        }

    }
}
