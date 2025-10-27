static void mb_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    MicroBlazeCPU *cpu = MICROBLAZE_CPU(obj);

    CPUMBState *env = &cpu->env;

    static bool tcg_initialized;



    cs->env_ptr = env;

    cpu_exec_init(cs, &error_abort);



    set_float_rounding_mode(float_round_nearest_even, &env->fp_status);



#ifndef CONFIG_USER_ONLY

    /* Inbound IRQ and FIR lines */

    qdev_init_gpio_in(DEVICE(cpu), microblaze_cpu_set_irq, 2);

#endif



    if (tcg_enabled() && !tcg_initialized) {

        tcg_initialized = true;

        mb_tcg_init();

    }

}
