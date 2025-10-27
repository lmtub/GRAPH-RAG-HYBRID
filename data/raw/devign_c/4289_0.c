static always_inline void gen_qemu_sts (TCGv t0, TCGv t1, int flags)

{

    TCGv tmp = tcg_temp_new(TCG_TYPE_I32);

    tcg_gen_helper_1_1(helper_s_to_memory, tmp, t0);

    tcg_gen_qemu_st32(tmp, t1, flags);

    tcg_temp_free(tmp);

}
