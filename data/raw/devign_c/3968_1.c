static void cris_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    CRISCPU *cpu = CRIS_CPU(obj);

    CRISCPUClass *ccc = CRIS_CPU_GET_CLASS(obj);

    CPUCRISState *env = &cpu->env;

    static bool tcg_initialized;



    cs->env_ptr = env;

    cpu_exec_init(cs, &error_abort);



    env->pregs[PR_VR] = ccc->vr;



#ifndef CONFIG_USER_ONLY

    /* IRQ and NMI lines.  */

    qdev_init_gpio_in(DEVICE(cpu), cris_cpu_set_irq, 2);

#endif



    if (tcg_enabled() && !tcg_initialized) {

        tcg_initialized = true;

        if (env->pregs[PR_VR] < 32) {

            cris_initialize_crisv10_tcg();

        } else {

            cris_initialize_tcg();

        }

    }

}
