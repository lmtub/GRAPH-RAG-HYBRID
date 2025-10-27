static void xtensa_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    XtensaCPU *cpu = XTENSA_CPU(obj);

    XtensaCPUClass *xcc = XTENSA_CPU_GET_CLASS(obj);

    CPUXtensaState *env = &cpu->env;

    static bool tcg_inited;



    cs->env_ptr = env;

    env->config = xcc->config;

    cpu_exec_init(cs, &error_abort);



    if (tcg_enabled() && !tcg_inited) {

        tcg_inited = true;

        xtensa_translate_init();

    }

}
