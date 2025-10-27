static void tilegx_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    TileGXCPU *cpu = TILEGX_CPU(obj);

    CPUTLGState *env = &cpu->env;

    static bool tcg_initialized;



    cs->env_ptr = env;

    cpu_exec_init(cs, &error_abort);



    if (tcg_enabled() && !tcg_initialized) {

        tcg_initialized = true;

        tilegx_tcg_init();

    }

}
