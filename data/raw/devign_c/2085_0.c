static void apic_startup(APICState *s, int vector_num)

{

    CPUState *env = s->cpu_env;

    if (!env->halted)

        return;

    env->eip = 0;

    cpu_x86_load_seg_cache(env, R_CS, vector_num << 8, vector_num << 12,

                           0xffff, 0);

    env->halted = 0;

}
