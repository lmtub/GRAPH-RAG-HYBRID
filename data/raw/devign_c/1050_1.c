static void superh_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    SuperHCPU *cpu = SUPERH_CPU(obj);

    CPUSH4State *env = &cpu->env;



    cs->env_ptr = env;

    cpu_exec_init(cs, &error_abort);



    env->movcal_backup_tail = &(env->movcal_backup);



    if (tcg_enabled()) {

        sh4_translate_init();

    }

}
