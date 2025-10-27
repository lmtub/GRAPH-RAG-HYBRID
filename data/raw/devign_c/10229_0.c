void cpu_exec_step_atomic(CPUState *cpu)

{

    start_exclusive();



    /* Since we got here, we know that parallel_cpus must be true.  */

    parallel_cpus = false;

    cpu_exec_step(cpu);

    parallel_cpus = true;



    end_exclusive();

}
