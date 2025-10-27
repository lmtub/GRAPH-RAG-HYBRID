static always_inline void gen_qemu_lds (TCGv t0, TCGv t1, int flags)

{

    TCGv tmp = tcg_temp_new(TCG_TYPE_I32);

    tcg_gen_qemu_ld32u(tmp, t1, flags);

    tcg_gen_helper_1_1(helper_memory_to_s, t0, tmp);

    tcg_temp_free(tmp);

}
