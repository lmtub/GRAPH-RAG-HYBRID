static void tricore_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    TriCoreCPU *cpu = TRICORE_CPU(obj);

    CPUTriCoreState *env = &cpu->env;



    cs->env_ptr = env;

    cpu_exec_init(cs, &error_abort);



    if (tcg_enabled()) {

        tricore_tcg_init();

    }

}
