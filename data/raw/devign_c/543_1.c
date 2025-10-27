void cpu_exit(CPUArchState *env)

{

    CPUState *cpu = ENV_GET_CPU(env);



    cpu->exit_request = 1;

    cpu_unlink_tb(cpu);

}
