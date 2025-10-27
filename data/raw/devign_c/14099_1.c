static void gen_dcbz(DisasContext *ctx)

{

    TCGv tcgv_addr;

    TCGv_i32 tcgv_is_dcbzl;

    int is_dcbzl = ctx->opcode & 0x00200000 ? 1 : 0;



    gen_set_access_type(ctx, ACCESS_CACHE);

    tcgv_addr = tcg_temp_new();

    tcgv_is_dcbzl = tcg_const_i32(is_dcbzl);



    gen_addr_reg_index(ctx, tcgv_addr);

    gen_helper_dcbz(cpu_env, tcgv_addr, tcgv_is_dcbzl);



    tcg_temp_free(tcgv_addr);

    tcg_temp_free_i32(tcgv_is_dcbzl);

}
