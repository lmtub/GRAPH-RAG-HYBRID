static void gen_st_cond (DisasContext *ctx, uint32_t opc, int rt,

                         int base, int16_t offset)

{

    const char *opn = "st_cond";

    TCGv t0, t1;



    t0 = tcg_temp_local_new();



    gen_base_offset_addr(ctx, t0, base, offset);

    /* Don't do NOP if destination is zero: we must perform the actual

       memory access. */



    t1 = tcg_temp_local_new();

    gen_load_gpr(t1, rt);

    switch (opc) {

#if defined(TARGET_MIPS64)

    case OPC_SCD:

        save_cpu_state(ctx, 0);

        op_st_scd(t1, t0, rt, ctx);

        opn = "scd";

        break;

#endif

    case OPC_SC:

        save_cpu_state(ctx, 1);

        op_st_sc(t1, t0, rt, ctx);

        opn = "sc";

        break;

    }

    (void)opn; /* avoid a compiler warning */

    MIPS_DEBUG("%s %s, %d(%s)", opn, regnames[rt], offset, regnames[base]);

    tcg_temp_free(t1);

    tcg_temp_free(t0);

}
