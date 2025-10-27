static void pc_cpu_reset(void *opaque)

{

    X86CPU *cpu = opaque;

    CPUX86State *env = &cpu->env;



    cpu_reset(CPU(cpu));

    env->halted = !cpu_is_bsp(env);

}
