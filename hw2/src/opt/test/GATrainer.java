package opt.test;

import opt.EvaluationFunction;
import opt.OptimizationAlgorithm;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.FixedIterationTrainer;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chanes on 3/9/17.
 */
public class GATrainer {

    private int populationSize;
    private int toMate;
    private int toMutate;
    private GeneticAlgorithmProblem gap;
    private EvaluationFunction ef;
    private int tValue;

    private ArrayList<Integer> iterations = new ArrayList<>();
    private ArrayList<Double> times = new ArrayList<>();
    private ArrayList<Double> opts = new ArrayList<>();
    private ArrayList<Double> errs = new ArrayList<>();

    public GATrainer(int populationSize, int toMate, int toMutate, GeneticAlgorithmProblem gap, EvaluationFunction ef, int tValue) {
        this.populationSize = populationSize;
        this.toMate = toMate;
        this.toMutate = toMutate;
        this.gap = gap;
        this.tValue = tValue;
        this.ef = ef;
    }

    public void train() {
        for(int i = 0; i < 20; i++) {
            double starttime = System.currentTimeMillis();
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, gap);
            FixedIterationTrainer trainer = new FixedIterationTrainer(ga, 5000);
            errs.add(trainer.train());
            this.opts.add(ef.value(ga.getOptimal()));
            this.times.add(System.currentTimeMillis() - starttime);
        }
        double optAvg = getDoubleAvg(this.opts);
        double timeAvg = getDoubleAvg(this.times);
        double errAvg = getDoubleAvg(this.errs);
        System.out.println(tValue + "\t" + optAvg + "\t" + errAvg + "\t" + timeAvg);
    }

    public void tune() {
        System.out.println("Tuning popSize");
        for (int p = 300; p <= 500; p = p + 20) {
            System.out.println("p: "+ p);
            for(int i = 0; i < 10; i++) {
                double starttime = System.currentTimeMillis();
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(p, toMate, toMutate, gap);
                FixedIterationTrainer trainer = new FixedIterationTrainer(ga, 5000);
                errs.add(trainer.train());
                this.opts.add(ef.value(ga.getOptimal()));
                this.times.add(System.currentTimeMillis() - starttime);
            }
            double optAvg = getDoubleAvg(this.opts);
            double timeAvg = getDoubleAvg(this.times);
            double errAvg = getDoubleAvg(this.errs);
            System.out.println(tValue + "\t" + optAvg + "\t" + errAvg + "\t" + timeAvg);
        this.opts.clear();
        this.times.clear();
        }

        System.out.println("Tuning toMate");
        for (int tm = 10; tm <= 250; tm = tm + 10) {
            System.out.println("tm: "+ tm);
            for(int i = 0; i < 10; i++) {
                double starttime = System.currentTimeMillis();
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(populationSize, tm, toMutate, gap);
                FixedIterationTrainer trainer = new FixedIterationTrainer(ga, 5000);
                errs.add(trainer.train());
                this.opts.add(ef.value(ga.getOptimal()));
                this.times.add(System.currentTimeMillis() - starttime);
            }
            double optAvg = getDoubleAvg(this.opts);
            double timeAvg = getDoubleAvg(this.times);
            double errAvg = getDoubleAvg(this.errs);
            System.out.println(tValue + "\t" + optAvg + "\t" + errAvg + "\t" + timeAvg);
            this.opts.clear();
            this.times.clear();
        }
        //use either 10 or 100

        System.out.println("Tuning toMutate");
        for (int tm = 5; tm <= 50; tm = tm + 5) {
            System.out.println("tm: "+ tm);
            for(int i = 0; i < 10; i++) {
                double starttime = System.currentTimeMillis();
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(populationSize, toMate, tm, gap);
                FixedIterationTrainer trainer = new FixedIterationTrainer(ga, 5000);
                errs.add(trainer.train());
                this.opts.add(ef.value(ga.getOptimal()));
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
