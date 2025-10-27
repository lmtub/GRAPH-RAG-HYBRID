static void init_proc_POWER9(CPUPPCState *env)
{
    /* Common Registers */
    init_proc_book3s_common(env);
    gen_spr_book3s_207_dbg(env);
    /* POWER8 Specific Registers */
    gen_spr_book3s_ids(env);
    gen_spr_amr(env);
    gen_spr_iamr(env);
    gen_spr_book3s_purr(env);
    gen_spr_power5p_common(env);
    gen_spr_power5p_lpar(env);
    gen_spr_power5p_ear(env);
    gen_spr_power6_common(env);
    gen_spr_power6_dbg(env);
    gen_spr_power8_tce_address_control(env);
    gen_spr_power8_ids(env);
    gen_spr_power8_ebb(env);
    gen_spr_power8_fscr(env);
    gen_spr_power8_pmu_sup(env);
    gen_spr_power8_pmu_user(env);
    gen_spr_power8_tm(env);
    gen_spr_power8_pspb(env);
    gen_spr_vtb(env);
    gen_spr_power8_ic(env);
    gen_spr_power8_book4(env);
    gen_spr_power8_rpr(env);
    /* POWER9 Specific registers */
    spr_register_kvm(env, SPR_TIDR, "TIDR", NULL, NULL,
                     KVM_REG_PPC_TIDR, 0);
    /* env variables */
#if !defined(CONFIG_USER_ONLY)
    env->slb_nr = 32;
#endif
    env->ci_large_pages = true;
    env->dcache_line_size = 128;
    env->icache_line_size = 128;
    /* Allocate hardware IRQ controller */
    init_excp_POWER8(env);
    ppcPOWER7_irq_init(ppc_env_get_cpu(env));
}