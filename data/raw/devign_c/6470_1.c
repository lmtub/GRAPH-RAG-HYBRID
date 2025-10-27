clk_setup_cb cpu_ppc_tb_init (CPUPPCState *env, uint32_t freq)

{

    PowerPCCPU *cpu = ppc_env_get_cpu(env);

    ppc_tb_t *tb_env;



    tb_env = g_malloc0(sizeof(ppc_tb_t));

    env->tb_env = tb_env;

    tb_env->flags = PPC_DECR_UNDERFLOW_TRIGGERED;

    if (env->insns_flags & PPC_SEGMENT_64B) {

        /* All Book3S 64bit CPUs implement level based DEC logic */

        tb_env->flags |= PPC_DECR_UNDERFLOW_LEVEL;

    }

    /* Create new timer */

    tb_env->decr_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, &cpu_ppc_decr_cb, cpu);

    if (0) {

        /* XXX: find a suitable condition to enable the hypervisor decrementer

         */

        tb_env->hdecr_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, &cpu_ppc_hdecr_cb,

                                                cpu);

    } else {

        tb_env->hdecr_timer = NULL;

    }

    cpu_ppc_set_tb_clk(env, freq);



    return &cpu_ppc_set_tb_clk;

}
