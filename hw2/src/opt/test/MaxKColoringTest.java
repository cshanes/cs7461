package opt.test;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.Distribution;
import opt.*;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.ConvergenceTrainer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 
 * @author kmandal
 * @version 1.0
 */
public class MaxKColoringTest {
    /** The n value */
    private static final int N = 500; // number of vertices
    private static final int L =4; // L adjacent nodes per vertex
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {

        System.out.println("k\tsaIter\tgaIter\tmimicIter\tsaTime\tgaTime\tmimicTime");

        for (int k = 2; k < 40; k = k + 2) {

            Random random = new Random(N*L);
            // create the random velocity
            Vertex[] vertices = new Vertex[N];
            for (int i = 0; i < N; i++) {
                Vertex vertex = new Vertex();
                vertices[i] = vertex;
                vertex.setAdjMatrixSize(L);
                for(int j = 0; j <L; j++ ){
                    vertex.getAadjacencyColorMatrix().add(random.nextInt(N*L));
                }
            }
            // for rhc, sa, and ga we use a permutation based encoding
            MaxKColorFitnessFunction ef = new MaxKColorFitnessFunction(vertices);
            Distribution odd = new DiscretePermutationDistribution(k);
            NeighborFunction nf = new SwapNeighbor();
            MutationFunction mf = new SwapMutation();
            CrossoverFunction cf = new SingleCrossOver();
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

            Distribution df = new DiscreteDependencyTree(.1);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            ArrayList<Integer> saIterations = new ArrayList<>();
            ArrayList<Integer> gaIterations = new ArrayList<>();
            ArrayList<Integer> mimicIterations = new ArrayList<>();
            ArrayList<Long> saTimes = new ArrayList<>();
            ArrayList<Long> gaTimes = new ArrayList<>();
            ArrayList<Long> mimicTimes = new ArrayList<>();
            for(int i = 0; i < 10; i++) {
                // SA
                long starttime = System.currentTimeMillis();
                SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .1, hcp);
                ConvergenceTrainer fit = new ConvergenceTrainer(sa);
                fit.train();
                saIterations.add(fit.getIterations());
                saTimes.add(System.currentTimeMillis() - starttime);

                // GA
                starttime = System.currentTimeMillis();
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 10, 60, gap);
                fit = new ConvergenceTrainer(ga);
                fit.train();
                gaIterations.add(fit.getIterations());
                gaTimes.add(System.currentTimeMillis() - starttime);

                // MIMIC
                starttime = System.currentTimeMillis();
                MIMIC mimic = new MIMIC(200, 100, pop);
                fit = new ConvergenceTrainer(mimic);
                fit.train();
                mimicIterations.add(fit.getIterations());
                mimicTimes.add(System.currentTimeMillis() - starttime);
            }
            double saIterAvg = getIntAvg(saIterations);
            double gaIterAvg = getIntAvg(gaIterations);
            double mimicIterAvg = getIntAvg(mimicIterations);
            double saTimeAvg = getAvg(saTimes);
            double gaTimeAvg = getAvg(gaTimes);
            double mimicTimeAvg = getAvg(mimicTimes);
            System.out.println(k + "\t" + saIterAvg + "\t" + gaIterAvg + "\t" + mimicIterAvg + "\t"
                    + saTimeAvg + "\t" + gaTimeAvg + "\t" + mimicTimeAvg);
        }
    }

    private static double getIntAvg(List<Integer> list) {
        return list.stream().mapToDouble(a -> a).average().getAsDouble();
    }

    private static double getAvg(List<Long> list) {
        return list.stream().mapToDouble(a -> a).average().getAsDouble();
    }
}
