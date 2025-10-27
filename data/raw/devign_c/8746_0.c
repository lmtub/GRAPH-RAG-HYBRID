static void gen_callwi(DisasContext *dc, int callinc, uint32_t dest, int slot)

{

    TCGv_i32 tmp = tcg_const_i32(dest);

    if (((dc->pc ^ dest) & TARGET_PAGE_MASK) != 0) {

        slot = -1;

    }

    gen_callw_slot(dc, callinc, tmp, slot);

    tcg_temp_free(tmp);

}
