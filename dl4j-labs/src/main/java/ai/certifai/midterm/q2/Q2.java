package ai.certifai.midterm.q2;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class Q2 {
    static int outputs = 10;
    static int batchSize = 128;
    static int seed = 123;
    static double learningRate = 0.0015;
    static int epochs = 10;
    static double splitRatio = 0.8;

    public static void main(String[] args) throws IOException, InterruptedException {
        File file = new ClassPathResource("mnist784/mnist_784.csv").getFile();
        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(new FileSplit(file));

        Schema schema = new Schema.Builder()
                .addColumnInteger("pixel%d", 1, 784)
                .build();

        System.out.println("record size: " + rr.next().size());

        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(rr, rr.next().size(), -1, outputs);
        DataSet fullData = dataSetIterator.next();

        fullData.shuffle(seed);

        SplitTestAndTrain testAndTrain = fullData.splitTestAndTrain(splitRatio);
        DataSet trainSet = testAndTrain.getTrain();
        DataSet testSet = testAndTrain.getTest();

        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(trainSet);
        normalizer.transform(trainSet);
        normalizer.transform(testSet);

        DataSetIterator trainIter = new ViewIterator(trainSet, batchSize);
        DataSetIterator testIter = new ViewIterator(testSet, batchSize);

        // TODO: Use Convolutional Neural Network (CNN)

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .nIn(1) // num of channels (grayscale = 1, rgb = 3)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .nOut(48)
                        .build())
                .layer(1, new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .padding(2, 2)
                        .activation(Activation.RELU)
                        .nOut(64)
                        .build())
                .layer(2, new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(3, new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .padding(1, 1)
                        .activation(Activation.RELU)
                        .nOut(128)
                        .build())
                .layer(4, new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(5, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(64)
                        .build())
                .layer(6, new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(outputs)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .backpropType(BackpropType.Standard)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

//        UIServer server = UIServer.getInstance();
//        StatsStorage storage = new InMemoryStatsStorage();
//        server.attach(storage);
//        model.setListeners(new StatsListener(storage, 10));

        model.setListeners(new ScoreIterationListener(100));

        // TODO: Add earlyStopping and K-Fold

        model.fit(trainIter, epochs);

        System.out.println("=== Train data evaluation ===");
        Evaluation trainEval = model.evaluate(trainIter);
        System.out.println(trainEval.stats());

        System.out.println("=== Valid data evaluation ===");
        Evaluation testEval = model.evaluate(trainIter);
        System.out.println(testEval.stats());
    }
}
