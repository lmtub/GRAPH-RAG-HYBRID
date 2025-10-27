void cpu_exit(CPUState *cpu)

{

    cpu->exit_request = 1;

    /* Ensure cpu_exec will see the exit request after TCG has exited.  */

    smp_wmb();

    cpu->tcg_exit_req = 1;

}
