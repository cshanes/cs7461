package opt.test;

import opt.EvaluationFunction;
import opt.OptimizationAlgorithm;
import shared.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chanes on 3/9/17.
 */
public class TestTrainer {

    private FixedIterationTrainer trainer;
    private EvaluationFunction ef;
    private int numIterations;
    private OptimizationAlgorithm algo;
    private double globalOptimum;
    private int tValue;

    private ArrayList<Integer> iterations = new ArrayList<>();
    private ArrayList<Double> times = new ArrayList<>();
    private ArrayList<Double> opts = new ArrayList<>();
    private ArrayList<Double> errs = new ArrayList<>();

    public TestTrainer(FixedIterationTrainer trainer, EvaluationFunction ef, OptimizationAlgorithm algo, int numIterations, double globalOptimum, int tValue) {
        this.trainer = trainer;
        this.ef = ef;
        this.algo = algo;
        this.numIterations = numIterations;
        this.globalOptimum = globalOptimum;
        this.tValue = tValue;
    }

    public void train() {
//        System.out.println("optAvg\terrAvg\titerAvg\ttimeAvg");
        for(int i = 0; i < numIterations; i++) {
            double starttime = System.currentTimeMillis();
            FixedIterationTrainer trainer2 = new FixedIterationTrainer(algo, trainer.getIterations());
            errs.add(trainer.train());
//            errs.add(trainer.train(globalOptimum, ef));
            this.opts.add(ef.value(algo.getOptimal()));
//            this.iterations.add(trainer.getIterations());
            this.times.add(System.currentTimeMillis() - starttime);
        }
        double optAvg = getDoubleAvg(this.opts);
//        double iterAvg = getIntAvg(this.iterations);
        double timeAvg = getDoubleAvg(this.times);
        double errAvg = getDoubleAvg(this.errs);
        System.out.println(tValue + "\t" + trainer.getIterations() + "\t" + optAvg + "\t" + errAvg +
                "\t" + timeAvg);
    }

    private static double getIntAvg(List<Integer> list) {
        return list.stream().mapToDouble(a -> a).average().getAsDouble();
    }

    private static double getDoubleAvg(List<Double> list) {
        return list.stream().mapToDouble(a -> a).average().getAsDouble();
    }
}
