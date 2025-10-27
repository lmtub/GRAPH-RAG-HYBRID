clk_setup_cb cpu_ppc_tb_init (CPUState *env, uint32_t freq)

{

    ppc_tb_t *tb_env;



    tb_env = qemu_mallocz(sizeof(ppc_tb_t));

    if (tb_env == NULL)

        return NULL;

    env->tb_env = tb_env;

    /* Create new timer */

    tb_env->decr_timer = qemu_new_timer(vm_clock, &cpu_ppc_decr_cb, env);

#if defined(TARGET_PPC64H)

    tb_env->hdecr_timer = qemu_new_timer(vm_clock, &cpu_ppc_hdecr_cb, env);

#endif /* defined(TARGET_PPC64H) */

    cpu_ppc_set_tb_clk(env, freq);



    return &cpu_ppc_set_tb_clk;

}
