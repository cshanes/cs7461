package opt.test;

import opt.EvaluationFunction;
import opt.HillClimbingProblem;
import opt.SimulatedAnnealing;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.FixedIterationTrainer;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chanes on 3/9/17.
 */
public class SATrainer {

    private double t;
    private double cooling;

    private HillClimbingProblem hcp;
    private EvaluationFunction ef;
    private int tValue;

    private ArrayList<Double> times = new ArrayList<>();
    private ArrayList<Double> opts = new ArrayList<>();
    private ArrayList<Double> errs = new ArrayList<>();

    public SATrainer(double t, double cooling, HillClimbingProblem hcp, EvaluationFunction ef, int tValue) {
        this.t = t;
        this.cooling = cooling;
        this.hcp = hcp;
        this.tValue = tValue;
        this.ef = ef;
    }

    public void train() {
        for(int i = 0; i < 20; i++) {
            double starttime = System.currentTimeMillis();
            SimulatedAnnealing sa = new SimulatedAnnealing(t, cooling, hcp);
            FixedIterationTrainer trainer = new FixedIterationTrainer(sa, 5000);
            errs.add(trainer.train());
            this.opts.add(ef.value(sa.getOptimal()));
            this.times.add(System.currentTimeMillis() - starttime);
        }
        double optAvg = getDoubleAvg(this.opts);
        double timeAvg = getDoubleAvg(this.times);
        double errAvg = getDoubleAvg(this.errs);
        System.out.println(tValue + "\t" + optAvg + "\t" + errAvg + "\t" + timeAvg);
    }

    public void tune() {
        System.out.println("tuning start temp");
        for(int t = 50; t < 1000; t = t + 10) {
            System.out.println("t: " + t);
            for(int i = 0; i < 20; i++) {
                double starttime = System.currentTimeMillis();
                SimulatedAnnealing sa = new SimulatedAnnealing(t, cooling, hcp);
                FixedIterationTrainer trainer = new FixedIterationTrainer(sa, 5000);
                errs.add(trainer.train());
                this.opts.add(ef.value(sa.getOptimal()));
                this.times.add(System.currentTimeMillis() - starttime);
            }
            double optAvg = getDoubleAvg(this.opts);
            double timeAvg = getDoubleAvg(this.times);
            double errAvg = getDoubleAvg(this.errs);
            System.out.println(tValue + "\t" + optAvg + "\t" + errAvg + "\t" + timeAvg);
            this.opts.clear();
            this.times.clear();
        }

        System.out.println("tuning cooling");
        for(double p = 0.05; p < 1.0; p = p + 0.05) {
            System.out.println("cooling: " + p);
            for(int i = 0; i < 20; i++) {
                double starttime = System.currentTimeMillis();
                SimulatedAnnealing sa = new SimulatedAnnealing(t, p, hcp);
                FixedIterationTrainer trainer = new FixedIterationTrainer(sa, 5000);
                errs.add(trainer.train());
                this.opts.add(ef.value(sa.getOptimal()));
                this.times.add(System.currentTimeMillis() - starttime);
            }
            double optAvg = getDoubleAvg(this.opts);
            double timeAvg = getDoubleAvg(this.times);
            double errAvg = getDoubleAvg(this.errs);
            System.out.println(tValue + "\t" + optAvg + "\t" + errAvg + "\t" + timeAvg);
            this.opts.clear();
            this.times.clear();
        }

    }

    private static double getIntAvg(List<Integer> list) {
        return list.stream().mapToDouble(a -> a).average().getAsDouble();
    }

    private static double getDoubleAvg(List<Double> list) {
        return list.stream().mapToDouble(a -> a).average().getAsDouble();
    }
}
