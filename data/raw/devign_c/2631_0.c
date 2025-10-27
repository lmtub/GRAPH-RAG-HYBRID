void slavio_intctl_set_cpu(void *opaque, unsigned int cpu, CPUState *env)

{

    SLAVIO_INTCTLState *s = opaque;



    s->cpu_envs[cpu] = env;

}
