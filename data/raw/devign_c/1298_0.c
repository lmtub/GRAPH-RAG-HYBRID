X86CPU *cpu_x86_init(const char *cpu_model)

{

    X86CPU *cpu;

    CPUX86State *env;

    static int inited;



    cpu = X86_CPU(object_new(TYPE_X86_CPU));

    env = &cpu->env;

    env->cpu_model_str = cpu_model;



    /* init various static tables used in TCG mode */

    if (tcg_enabled() && !inited) {

        inited = 1;

        optimize_flags_init();

#ifndef CONFIG_USER_ONLY

        prev_debug_excp_handler =

            cpu_set_debug_excp_handler(breakpoint_handler);

#endif

    }

    if (cpu_x86_register(cpu, cpu_model) < 0) {

        object_delete(OBJECT(cpu));

        return NULL;

    }



    x86_cpu_realize(OBJECT(cpu), NULL);



    return cpu;

}
