static always_inline void gen_qemu_ld32s(DisasContext *ctx, TCGv arg1, TCGv arg2)

{

    if (unlikely(ctx->mem_idx)) {

        TCGv_i32 t0;

        tcg_gen_qemu_ld32u(arg1, arg2, ctx->mem_idx);

        t0 = tcg_temp_new_i32();

        tcg_gen_trunc_tl_i32(t0, arg1);

        tcg_gen_bswap_i32(t0, t0);

        tcg_gen_ext_i32_tl(arg1, t0);

        tcg_temp_free_i32(t0);

    } else

        tcg_gen_qemu_ld32s(arg1, arg2, ctx->mem_idx);

}
