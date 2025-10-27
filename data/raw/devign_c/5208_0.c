static void init_proc_book3s_64(CPUPPCState *env, int version)

{

    gen_spr_ne_601(env);

    gen_tbl(env);

    gen_spr_book3s_altivec(env);

    gen_spr_book3s_pmu_sup(env);

    gen_spr_book3s_pmu_user(env);

    gen_spr_book3s_dbg(env);

    gen_spr_book3s_common(env);



    switch (version) {

    case BOOK3S_CPU_970:

    case BOOK3S_CPU_POWER5PLUS:

        gen_spr_970_hid(env);

        gen_spr_970_hior(env);

        gen_low_BATs(env);

        gen_spr_970_pmu_sup(env);

        gen_spr_970_pmu_user(env);

        break;

    case BOOK3S_CPU_POWER7:

    case BOOK3S_CPU_POWER8:

        gen_spr_book3s_ids(env);

        gen_spr_amr(env);

        gen_spr_book3s_purr(env);

        break;

    default:

        g_assert_not_reached();

    }

    if (version >= BOOK3S_CPU_POWER5PLUS) {

        gen_spr_power5p_common(env);

        gen_spr_power5p_lpar(env);

        gen_spr_power5p_ear(env);

    } else {

        gen_spr_970_lpar(env);

    }

    if (version == BOOK3S_CPU_970) {

        gen_spr_970_dbg(env);

    }

    if (version >= BOOK3S_CPU_POWER6) {

        gen_spr_power6_common(env);

        gen_spr_power6_dbg(env);

    }

    if (version >= BOOK3S_CPU_POWER8) {

        gen_spr_power8_tce_address_control(env);

        gen_spr_power8_ids(env);

        gen_spr_power8_ebb(env);

        gen_spr_power8_fscr(env);

        gen_spr_power8_pmu_sup(env);

        gen_spr_power8_pmu_user(env);

        gen_spr_power8_tm(env);

    }

#if !defined(CONFIG_USER_ONLY)

    switch (version) {

    case BOOK3S_CPU_970:

    case BOOK3S_CPU_POWER5PLUS:

        env->slb_nr = 64;

        break;

    case BOOK3S_CPU_POWER7:

    case BOOK3S_CPU_POWER8:

    default:

        env->slb_nr = 32;

        break;

    }

#endif

    /* Allocate hardware IRQ controller */

    switch (version) {

    case BOOK3S_CPU_970:

    case BOOK3S_CPU_POWER5PLUS:

        init_excp_970(env);

        ppc970_irq_init(env);

        break;

    case BOOK3S_CPU_POWER7:

    case BOOK3S_CPU_POWER8:

        init_excp_POWER7(env);

        ppcPOWER7_irq_init(env);

        break;

    default:

        g_assert_not_reached();

    }



    env->dcache_line_size = 128;

    env->icache_line_size = 128;

}
