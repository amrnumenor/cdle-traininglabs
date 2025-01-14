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

package ai.certifai.training.datavec.kfold;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.StringColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.jfree.data.io.CSV;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class KFoldExercise {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";
    private static double learningRate = 0.001;
    private static int epochs = 100;
    private static int outcomes = 2;
    private static int label = 47;
    private static int kFold = 5;
    private static int numIn = 47;

    public static void main(String[] args) throws IOException, InterruptedException {


        //define the location of the file
        File file = new ClassPathResource("CreditCardApproval/uci_credit_approval.csv").getFile();

        //create record reader and initialize it
        RecordReader CSVReader = new CSVRecordReader(1);
        CSVReader.initialize(new FileSplit(file));

        // create schema
        Schema schema = new Schema.Builder()
                .addColumnsString("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16")
                .build();

        // create transform process based on the schema
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .filter(new ConditionFilter(
                        new StringColumnCondition("A1", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A1", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A1", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A2", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A3", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A4", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A5", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A6", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A7", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A8", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A9", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A10", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A11", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A12", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A13", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A14", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A15", ConditionOp.Equal, "?")))
                .filter(new ConditionFilter(
                        new StringColumnCondition("A16", ConditionOp.Equal, "?")))
                .stringToCategorical("A1", Arrays.asList("b", "a"))
                .stringToCategorical("A4", Arrays.asList("u", "y", "l", "t"))
                .stringToCategorical("A5", Arrays.asList("g", "p", "gg"))
                .stringToCategorical("A6", Arrays.asList("c", "d", "cc", "i", "j", "k",
                        "m", "r", "q", "w", "x", "e", "aa", "ff"))
                .stringToCategorical("A7", Arrays.asList("v", "h", "bb", "j", "n", "z", "dd", "ff", "o"))
                .stringToCategorical("A9", Arrays.asList("t", "f"))
                .stringToCategorical("A10", Arrays.asList("t", "f"))
                .stringToCategorical("A12", Arrays.asList("t", "f"))
                .stringToCategorical("A13", Arrays.asList("g", "p", "s"))
                .stringToCategorical("A16", Arrays.asList("+", "-"))
                .categoricalToOneHot("A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13")
                .categoricalToInteger("A16")
                .build();

        Schema finalSchema = transformProcess.getFinalSchema();
        System.out.println("Schema: ");
        System.out.println(finalSchema);

        //transform the data via transform process
        List<List<Writable>> originalData = new ArrayList<>();
        while(CSVReader.hasNext()) {
            originalData.add(CSVReader.next());
        }
        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, transformProcess);

        //show the info (optional)
        int numOfRows = processedData.size();
        System.out.println(processedData);
        System.out.println("Total number of rows: " + numOfRows);

        //create record reader and load in all data into a single batch
        CollectionRecordReader crr = new CollectionRecordReader(processedData);
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(crr, numOfRows, label, outcomes);
        DataSet allData = dataSetIterator.next();

        //shuffle the data
        allData.shuffle();

        //create kfold iterator (k=5)
        KFoldIterator kFoldIterator = new KFoldIterator(kFold, allData);

        //create neural network config
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numIn)
                        .nOut(100)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .nIn(100)
                        .nOut(outcomes)
                        .build())
                .build();

        //start the kfold evaluation
        int i = 1;
        System.out.println("-------------------------------------------------------------");

        //initialize an empty list to store the F1 score
        ArrayList<Double> f1Score = new ArrayList<>();

        //for each fold
        while(kFoldIterator.hasNext()) {
            System.out.println(BLACK_BOLD + "Fold: " + i + ANSI_RESET);
            //for each fold, get the features and labels from training set and test set
            DataSet currDataset = kFoldIterator.next();
            INDArray trainFoldFeatures = currDataset.getFeatures();
            INDArray trainFoldLabels = currDataset.getLabels();
            INDArray testFoldFeatures = kFoldIterator.testFold().getFeatures();
            INDArray testFoldLabels = kFoldIterator.testFold().getLabels();
            DataSet trainDataSet = new DataSet(trainFoldFeatures, trainFoldLabels);

            //scale the dataset
            DataNormalization scaler = new NormalizerMinMaxScaler();
            scaler.fit(trainDataSet);
            scaler.transform(trainFoldFeatures);
            scaler.transform(testFoldFeatures);

            //initialize the model
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

            //train the data
            for(int j = 1; j <= epochs; ++j) {
                model.fit(trainDataSet);
            }

            //evaluate the model with test set
            Evaluation eval = new Evaluation();
            eval.eval(testFoldLabels, model.output(testFoldFeatures));

            //print out the evaluation results
            System.out.println(eval.stats());

            //save the eval results
            f1Score.add(eval.f1());

            i++;
            System.out.println("-------------------------------------------------------------");
        }
        INDArray f1scores = Nd4j.create(f1Score);
        System.out.println("Average F1 scores for all folds: " + f1scores.mean(0));
    }

}