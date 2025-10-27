void cpu_interrupt(CPUArchState *env, int mask)

{

    CPUState *cpu = ENV_GET_CPU(env);



    env->interrupt_request |= mask;

    cpu_unlink_tb(cpu);

}
