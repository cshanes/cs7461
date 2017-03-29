package opt.test;

import opt.EvaluationFunction;
import opt.HillClimbingProblem;
import opt.SimulatedAnnealing;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chanes on 3/9/17.
 */
public class MIMICTrainer {

    private int sampleSize;
    private int toKeep;

    private ProbabilisticOptimizationProblem pop;
    private EvaluationFunction ef;
    private int tValue;

    private ArrayList<Double> times = new ArrayList<>();
    private ArrayList<Double> opts = new ArrayList<>();
    private ArrayList<Double> errs = new ArrayList<>();

    public MIMICTrainer(int sampleSize, int toKeep, ProbabilisticOptimizationProblem pop, EvaluationFunction ef, int tValue) {
        this.sampleSize = sampleSize;
        this.toKeep = toKeep;
        this.pop = pop;
        this.tValue = tValue;
        this.ef = ef;
    }

    public void train() {
        for(int i = 0; i < 5; i++) {
            double starttime = System.currentTimeMillis();
            MIMIC m = new MIMIC(sampleSize, toKeep, pop);
            FixedIterationTrainer trainer = new FixedIterationTrainer(m, 5000);
            errs.add(trainer.train());
            this.opts.add(ef.value(m.getOptimal()));
            this.times.add(System.currentTimeMillis() - starttime);
        }
        double optAvg = getDoubleAvg(this.opts);
        double timeAvg = getDoubleAvg(this.times);
        double errAvg = getDoubleAvg(this.errs);
        System.out.println(tValue + "\t" + optAvg + "\t" + errAvg + "\t" + timeAvg);
    }

    public void tune() {
        System.out.println("Tuning sample size");
        for(int s = 240; s <= 360; s = s + 20) {
            System.out.println("s: "+ s);
            for(int i = 0; i < 5; i++) {
                double starttime = System.currentTimeMillis();
                MIMIC m = new MIMIC(s, toKeep, pop);
                FixedIterationTrainer trainer = new FixedIterationTrainer(m, 5000);
                errs.add(trainer.train());
                this.opts.add(ef.value(m.getOptimal()));
                this.times.add(System.currentTimeMillis() - starttime);
            }
            double optAvg = getDoubleAvg(this.opts);
            double timeAvg = getDoubleAvg(this.times);
            double errAvg = getDoubleAvg(this.errs);
            System.out.println(tValue + "\t" + optAvg + "\t" + errAvg + "\t" + timeAvg);
        }

        System.out.println("Tuning tk");
        for (int tk = 5; tk <= 50; tk = tk + 5) {
            System.out.println("tk: "+ tk);
            for(int i = 0; i < 5; i++) {
                double starttime = System.currentTimeMillis();
                MIMIC m = new MIMIC(360, tk, pop);
                FixedIterationTrainer trainer = new FixedIterationTrainer(m, 5000);
                errs.add(trainer.train());
                this.opts.add(ef.value(m.getOptimal()));
                this.times.add(System.currentTimeMillis() - starttime);
            }
            double optAvg = getDoubleAvg(this.opts);
            double timeAvg = getDoubleAvg(this.times);
            double errAvg = getDoubleAvg(this.errs);
            System.out.println(tValue + "\t" + optAvg + "\t" + errAvg + "\t" + timeAvg);
        }

    }

    private static double getIntAvg(List<Integer> list) {
        return list.stream().mapToDouble(a -> a).average().getAsDouble();
    }

    private static double getDoubleAvg(List<Double> list) {
        return list.stream().mapToDouble(a -> a).average().getAsDouble();
    }
}
