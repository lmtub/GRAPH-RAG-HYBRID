static void gen_rdhwr(DisasContext *ctx, int rt, int rd, int sel)

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

        gen_helper_rdhwr_cpunum(t0, cpu_env);

        gen_store_gpr(t0, rt);

        break;

    case 1:

        gen_helper_rdhwr_synci_step(t0, cpu_env);

        gen_store_gpr(t0, rt);

        break;

    case 2:

        if (ctx->tb->cflags & CF_USE_ICOUNT) {

            gen_io_start();

        }

        gen_helper_rdhwr_cc(t0, cpu_env);

        if (ctx->tb->cflags & CF_USE_ICOUNT) {

            gen_io_end();

        }

        gen_store_gpr(t0, rt);

        /* Break the TB to be able to take timer interrupts immediately

           after reading count. BS_STOP isn't sufficient, we need to ensure

           we break completely out of translated code.  */

        gen_save_pc(ctx->pc + 4);

        ctx->bstate = BS_EXCP;

        break;

    case 3:

        gen_helper_rdhwr_ccres(t0, cpu_env);

        gen_store_gpr(t0, rt);

        break;

    case 4:

        check_insn(ctx, ISA_MIPS32R6);

        if (sel != 0) {

            /* Performance counter registers are not implemented other than

             * control register 0.

             */

            generate_exception(ctx, EXCP_RI);

        }

        gen_helper_rdhwr_performance(t0, cpu_env);

        gen_store_gpr(t0, rt);

        break;

    case 5:

        check_insn(ctx, ISA_MIPS32R6);

        gen_helper_rdhwr_xnp(t0, cpu_env);

        gen_store_gpr(t0, rt);

        break;

    case 29:

#if defined(CONFIG_USER_ONLY)

        tcg_gen_ld_tl(t0, cpu_env,

                      offsetof(CPUMIPSState, active_tc.CP0_UserLocal));

        gen_store_gpr(t0, rt);

        break;

#else

        if ((ctx->hflags & MIPS_HFLAG_CP0) ||

            (ctx->hflags & MIPS_HFLAG_HWRENA_ULR)) {

            tcg_gen_ld_tl(t0, cpu_env,

                          offsetof(CPUMIPSState, active_tc.CP0_UserLocal));

            gen_store_gpr(t0, rt);

        } else {

            generate_exception_end(ctx, EXCP_RI);

        }

        break;

#endif

    default:            /* Invalid */

        MIPS_INVAL("rdhwr");

        generate_exception_end(ctx, EXCP_RI);

        break;

    }

    tcg_temp_free(t0);

}
