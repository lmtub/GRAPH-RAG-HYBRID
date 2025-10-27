TCGv_i32 tcg_global_reg_new_i32(int reg, const char *name)

{

    int idx;



    idx = tcg_global_reg_new_internal(TCG_TYPE_I32, reg, name);

    return MAKE_TCGV_I32(idx);

}
