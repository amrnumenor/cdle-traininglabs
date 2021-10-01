/*
 * Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */
package ai.certifai.training.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;


import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
/***
 * Dataset
 * https://www.kaggle.com/zhangjuefei/birds-bones-and-living-habits
 *
 * @author BoonKhai Yeoh
 */

public class BirdClassification {
    static int seed = 123;
    static int numInput = 10;
    static int numClass = 6;
    static int epoch = 1000;
    static double splitRatio = 0.8;
    static double learningRate = 1e-2;
    static INDArray weightsArray = Nd4j.create(new double[]{0.4, 0.5, 1.0, 0.5, 0.5, 0.4});



    public static void main(String[] args) throws Exception{

        //set filepath
        File dataFile = new ClassPathResource("/birdclassify/bird.csv").getFile();

        //File split
        FileSplit fileSplit = new FileSplit(dataFile);


        //set CSV Record Reader and initialize it
        RecordReader rr = new CSVRecordReader(1,',');
        rr.initialize(fileSplit);

//=========================================================================
        //  Step 1 : Build Schema to prepare the data
//=========================================================================

        Schema sc = new Schema.Builder()
                .addColumnInteger("id")
                .addColumnsFloat("huml", "humw", "ulnal", "ulnaw", "feml", "femw", "tibl", "tibw", "tarl", "tarw")
                .addColumnCategorical("type", Arrays.asList("SW", "W", "T", "R", "P", "SO"))
                .build();


//=========================================================================
        //  Step 2 : Build TransformProcess to transform the data
//=========================================================================
        TransformProcess tp = new TransformProcess.Builder(sc)
                .removeColumns("id")
                .categoricalToInteger("type")
                .build();

//        Checking the schema
        Schema finalSchema = tp.getFinalSchema();
        System.out.println("Schema: \n" + finalSchema);

        List<List<Writable>> allData = new ArrayList<>();
        while(rr.hasNext()){
            allData.add(rr.next());
        }
        List<List<Writable>> processedData = LocalTransformExecutor.execute(allData, tp);

//========================================================================
        //  Step 3 : Create Iterator ,splitting trainData and testData
//========================================================================

        //Create iterator from process data
        CollectionRecordReader crr = new CollectionRecordReader(processedData);

        //Input batch size , label index , and number of label
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(crr, processedData.size(), 10, numClass);

        //Create Iterator and shuffle the dat
        DataSet fullData = dataSetIterator.next();
        fullData.shuffle();

        //Input split ratio
        SplitTestAndTrain testAndTrain = fullData.splitTestAndTrain(splitRatio);

        //Get train and test dataset
        DataSet trainData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //printout size ( uncomment these lines )
        System.out.println("Training vector : ");
        System.out.println(Arrays.toString(trainData.getFeatures().shape()));
        System.out.println("Test vector : ");
        System.out.println(Arrays.toString(testData.getFeatures().shape()));

        // Train and Test iterator
//        DataSetIterator trainIter = new ViewIterator(trainData, batchSize);
//        DataSetIterator testIter = new ViewIterator(testData, batchSize);

//========================================================================
        //  Step 4 : DataNormalization
//========================================================================

        //Data normalization
        DataNormalization scaler = new NormalizerMinMaxScaler();
        scaler.fit(trainData);
        scaler.transform(trainData);
        scaler.transform(testData);

//========================================================================
        //  Step 5 : Network Configuration
//========================================================================

        //Get network configuration ( uncomment these lines )
        MultiLayerConfiguration config = getConfig(numInput, numClass, learningRate);

        //Define network ( uncomment these lines )
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

//========================================================================
        //  Step 6 : Setup UI , listeners
//========================================================================

        //UI-Evaluator
        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);

        //Set model listeners ( uncomment these lines )
        model.setListeners(new StatsListener(storage, 10));

//========================================================================
        //  Step 7 : Training
//========================================================================

        //Training
        Evaluation eval;
        for (int i = 0; i < epoch; ++i) {
            model.fit(trainData);
            eval = model.evaluate(new ViewIterator(testData, processedData.size()));
            System.out.println("EPOCH: " + i + " Accuracy: " + eval.accuracy());
        }

//========================================================================
        //  Step 8 : Evaluation
//========================================================================

        //Confusion matrix

        //TrainData
        Evaluation evalTrain = model.evaluate(new ViewIterator(trainData, processedData.size()));
        System.out.println("Train Data");
        System.out.println(evalTrain.stats());

        //TestData
        Evaluation evalTest = model.evaluate(new ViewIterator(testData, processedData.size()));
        System.out.println("Test Data");
        System.out.println(evalTest.stats());

//========================================================================
        //  Step 9 : Save model
//========================================================================
        File location = new File(System.getProperty("java.io.tmpdir"), "/trained_birdClass_model.zip");
        System.out.println("Saved model at: " + location);
        ModelSerializer.writeModel(model, location, true);

    }
    public static MultiLayerConfiguration getConfig(int numInputs, int numOutputs, double learningRate) {

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .l2(0.001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(40)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(40)
                        .nOut(30)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(30)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(20)
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new OutputLayer.Builder()
                        .nIn(10)
                        .nOut(numOutputs)
                        .lossFunction(new LossMCXENT(weightsArray))
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        return config;
    }
}
