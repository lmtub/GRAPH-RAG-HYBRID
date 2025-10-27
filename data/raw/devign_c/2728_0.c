static void cpu_4xx_pit_cb (void *opaque)

{

    CPUState *env;

    ppc_tb_t *tb_env;

    ppcemb_timer_t *ppcemb_timer;



    env = opaque;

    tb_env = env->tb_env;

    ppcemb_timer = tb_env->opaque;

    env->spr[SPR_40x_TSR] |= 1 << 27;

    if ((env->spr[SPR_40x_TCR] >> 26) & 0x1)

        ppc_set_irq(env, PPC_INTERRUPT_PIT, 1);

    start_stop_pit(env, tb_env, 1);

    LOG_TB("%s: ar %d ir %d TCR " TARGET_FMT_lx " TSR " TARGET_FMT_lx " "

           "%016" PRIx64 "\n", __func__,

           (int)((env->spr[SPR_40x_TCR] >> 22) & 0x1),

           (int)((env->spr[SPR_40x_TCR] >> 26) & 0x1),

           env->spr[SPR_40x_TCR], env->spr[SPR_40x_TSR],

           ppcemb_timer->pit_reload);

}
