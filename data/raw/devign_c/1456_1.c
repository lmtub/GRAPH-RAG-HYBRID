static void mips_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    MIPSCPU *cpu = MIPS_CPU(obj);

    CPUMIPSState *env = &cpu->env;



    cs->env_ptr = env;

    cpu_exec_init(cs, &error_abort);



    if (tcg_enabled()) {

        mips_tcg_init();

    }

}
