static void gen_rdhwr(DisasContext *ctx, int rt, int rd)

{

    TCGv t0;



#if !defined(CONFIG_USER_ONLY)

    /* The Linux kernel will emulate rdhwr if it's not supported natively.

       Therefore only check the ISA in system mode.  */

    check_insn(ctx, ISA_MIPS32R2);

#endif

    t0 = tcg_temp_new();



    switch (rd) {

    case 0:

        save_cpu_state(ctx, 1);

        gen_helper_rdhwr_cpunum(t0, cpu_env);

        gen_store_gpr(t0, rt);

        break;

    case 1:

        save_cpu_state(ctx, 1);

        gen_helper_rdhwr_synci_step(t0, cpu_env);

        gen_store_gpr(t0, rt);

        break;

    case 2:

        save_cpu_state(ctx, 1);

        gen_helper_rdhwr_cc(t0, cpu_env);

        gen_store_gpr(t0, rt);

        break;

    case 3:

        save_cpu_state(ctx, 1);

        gen_helper_rdhwr_ccres(t0, cpu_env);

        gen_store_gpr(t0, rt);

        break;

    case 29:

#if defined(CONFIG_USER_ONLY)

        tcg_gen_ld_tl(t0, cpu_env, offsetof(CPUMIPSState, tls_value));

        gen_store_gpr(t0, rt);

        break;

#else

        /* XXX: Some CPUs implement this in hardware.

           Not supported yet. */

#endif

    default:            /* Invalid */

        MIPS_INVAL("rdhwr");

        generate_exception(ctx, EXCP_RI);

        break;

    }

    tcg_temp_free(t0);

}
