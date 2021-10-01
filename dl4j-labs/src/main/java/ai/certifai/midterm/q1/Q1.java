package ai.certifai.midterm.q1;

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
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Q1 {
    static int epochs = 200;
    static int seed = 123;
    static int batchSize = 256;
    static double learningRate = 0.0015;
    static double splitRatio = 0.8;

    public static void main(String[] args) throws IOException, InterruptedException {

        // Load File
        File trainFile = new ClassPathResource("question1/train.csv").getFile();
        RecordReader trainRecord = new CSVRecordReader(1, ',');
        trainRecord.initialize(new FileSplit(trainFile));


        Schema schema = new Schema.Builder()
                .addColumnsInteger("id", "age")
                .addColumnString("job")
                .addColumnCategorical("marital", Arrays.asList("married", "divorced", "single"))
                .addColumnCategorical("education", Arrays.asList("primary", "secondary", "tertiary", "unknown"))
                .addColumnCategorical("default", Arrays.asList("yes", "no"))
                .addColumnInteger("balance")
                .addColumnCategorical("housing", "yes", "no")
                .addColumnCategorical("loan", "yes", "no")
                .addColumnCategorical("contact", Arrays.asList("telephone", "cellular", "unknown"))
                .addColumnInteger("day")
                .addColumnString("month")
                .addColumnsInteger("duration", "campaign", "pdays", "previous")
                .addColumnCategorical("poutcome", Arrays.asList("unknown", "success", "failure", "other"))
                .addColumnCategorical("subscribed", "yes", "no")
                .build();

        System.out.println("Initial Schema: \n" + schema);

        TransformProcess tp = new TransformProcess.Builder(schema)
                .removeColumns("id", "day", "month", "pdays", "poutcome")
                .stringMapTransform("job", Collections.singletonMap("admin.", "admin"))
                .stringToCategorical("job", Arrays.asList(
                        "admin",
                        "services",
                        "management",
                        "technician",
                        "blue-collar",
                        "housemaid",
                        "retired",
                        "student",
                        "entrepreneur",
                        "self-employed",
                        "unemployed",
                        "unknown"))
                .categoricalToOneHot("job", "housing", "loan", "default", "contact")
                .categoricalToInteger("education", "marital", "subscribed")
                .build();

        System.out.println("FinalSchema: \n" + tp.getFinalSchema());

        List<List<Writable>> allTrain = new ArrayList<>();
        while(trainRecord.hasNext())
            allTrain.add(trainRecord.next());
        List<List<Writable>> processTrain = LocalTransformExecutor.execute(allTrain, tp);

        CollectionRecordReader crr = new CollectionRecordReader(processTrain);
        System.out.println("process train size: " + processTrain.size());
        DataSetIterator trainDatasetIterator = new RecordReaderDataSetIterator(crr, processTrain.size(), -1, 2);

        DataSet fullTrain = trainDatasetIterator.next();
        fullTrain.shuffle(seed);

        SplitTestAndTrain trainAndValid = fullTrain.splitTestAndTrain(splitRatio);
        DataSet trainData = trainAndValid.getTrain();
        DataSet validData = trainAndValid.getTest();

        System.out.println("Training vector: ");
        System.out.println(Arrays.toString(trainData.getFeatures().shape()));
        System.out.println("Validation vector: ");
        System.out.println(Arrays.toString(validData.getFeatures().shape()));

        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(trainData);
        normalizer.transform(trainData);
        normalizer.transform(validData);

        ViewIterator trainIter = new ViewIterator(trainData, batchSize);
        ViewIterator validIter = new ViewIterator(validData, batchSize);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .l2(0.001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(trainData.numInputs())
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(256)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new OutputLayer.Builder()
                        .nIn(64)
                        .lossFunction(new LossBinaryXENT(Nd4j.create(new double[]{0.8, 0.5})))
                        .activation(Activation.SIGMOID)
                        .nOut(trainData.numOutcomes())
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

//        UIServer server = UIServer.getInstance();
//        StatsStorage storage = new InMemoryStatsStorage();
//        server.attach(storage);
//        model.setListeners(new StatsListener(storage, 10));

        model.setListeners(new ScoreIterationListener(100));

        // TODO: Add earlyStopping and K-Fold

        Evaluation eval;
        for(int i = 1; i <= epochs; ++i) {
            model.fit(trainIter);
            eval = model.evaluate(validIter);
            System.out.println("Epochs: " + i + " Val Accuracy: " + eval.accuracy());
        }

        System.out.println("=== Train data evaluation ===");
        Evaluation trainEval = model.evaluate(trainIter);
        System.out.println(trainEval.stats());

        System.out.println("=== Valid data evaluation ===");
        Evaluation validEval = model.evaluate(validIter);
        System.out.println(validEval.stats());

        /* TODO: Save Model */
        // ModelSerializer.writeModel();

        /* TODO: Test Data */
        // model.output()
    }
}

