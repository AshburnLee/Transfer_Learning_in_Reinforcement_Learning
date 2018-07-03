package myProj;

import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeTF;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
import burlap.domain.singleagent.blockdude.state.BlockDudeAgent;
import burlap.domain.singleagent.blockdude.state.BlockDudeCell;
import burlap.domain.singleagent.blockdude.state.BlockDudeMap;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor;


public class JunhuiBlockDude {
    
    BlockDude bd;
    OOSADomain domain;
    HashableStateFactory hashFactory;
    SimulatedEnvironment env;
    TerminalFunction tf;
    State initialState;
    StateConditionTest goalCondition;

    /* Constructer */
    public JunhuiBlockDude(){
        bd = new BlockDude();
        bd = new BlockDude(10,10);

        domain = bd.generateDomain();

	//  create my own environment
        initialState = new BlockDudeState(new BlockDudeAgent(1,1,0,false),
                new BlockDudeMap(new int[][] {
                        {1,1,1,0,0,0,0,0,0,0},
                        {1,0,0,0,0,0,0,0,0,0},
                        {1,0,0,0,0,0,0,0,0,0},
                        {1,1,0,0,0,0,0,0,0,0},
                        {1,1,1,0,0,0,0,0,0,0},
                        {1,1,0,0,0,0,0,0,0,0},
                        {1,0,0,0,0,0,0,0,0,0},
                        {1,0,0,0,0,0,0,0,0,0},
                        {1,0,0,0,0,0,0,0,0,0},
                        {1,1,1,1,1,1,1,0,0,0}}),
                new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "Exit"));

        hashFactory = new SimpleHashableStateFactory();
        env = new SimulatedEnvironment(domain, initialState);

    }
    /* Q learning implementation */
    public void qLearning(String output) {
        LearningAgent agent = new QLearning(domain, 0.99, hashFactory, 0.0, 1.0);

        //run learning 50 episodes
        for (int i=0;i<20;i++){
            Episode e = agent.runLearningEpisode(env);
            e.write(output + "QL" + i);
            System.out.println(i + ": " + e.maxTimeStep());
            // reset env for the next episode
            env.resetEnvironment();
        }
    }
    /* Visualize the learning process */
    public void visualize(String output){
        Visualizer v = BlockDudeVisualizer.getVisualizer(10, 10);
        new EpisodeSequenceVisualizer(v, domain, output);
    }


    /* Entry */
    public static void main(String[] args){

        System.out.println("check points...");
        String output = "JunhuiBD_output/";
        JunhuiBlockDude example = new JunhuiBlockDude();

        example.qLearning(output);
        example.visualize(output);
        return;

    }

}
