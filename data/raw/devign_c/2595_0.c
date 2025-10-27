static void gen_jumpi(DisasContext *dc, uint32_t dest, int slot)

{

    TCGv_i32 tmp = tcg_const_i32(dest);

    if (((dc->pc ^ dest) & TARGET_PAGE_MASK) != 0) {

        slot = -1;

    }

    gen_jump_slot(dc, tmp, slot);

    tcg_temp_free(tmp);

}
