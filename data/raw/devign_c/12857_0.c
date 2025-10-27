static void gen_rlwnm(DisasContext *ctx)

{

    uint32_t mb, me;

    mb = MB(ctx->opcode);

    me = ME(ctx->opcode);



    if (likely(mb == 0 && me == 31)) {

        TCGv_i32 t0, t1;

        t0 = tcg_temp_new_i32();

        t1 = tcg_temp_new_i32();

        tcg_gen_trunc_tl_i32(t0, cpu_gpr[rB(ctx->opcode)]);

        tcg_gen_trunc_tl_i32(t1, cpu_gpr[rS(ctx->opcode)]);

        tcg_gen_andi_i32(t0, t0, 0x1f);

        tcg_gen_rotl_i32(t1, t1, t0);

        tcg_gen_extu_i32_tl(cpu_gpr[rA(ctx->opcode)], t1);

        tcg_temp_free_i32(t0);

        tcg_temp_free_i32(t1);

    } else {

        TCGv t0;

#if defined(TARGET_PPC64)

        TCGv t1;

#endif



        t0 = tcg_temp_new();

        tcg_gen_andi_tl(t0, cpu_gpr[rB(ctx->opcode)], 0x1f);

#if defined(TARGET_PPC64)

        t1 = tcg_temp_new_i64();

        tcg_gen_deposit_i64(t1, cpu_gpr[rS(ctx->opcode)],

                            cpu_gpr[rS(ctx->opcode)], 32, 32);

        tcg_gen_rotl_i64(t0, t1, t0);

        tcg_temp_free_i64(t1);

#else

        tcg_gen_rotl_i32(t0, cpu_gpr[rS(ctx->opcode)], t0);

#endif

        if (unlikely(mb != 0 || me != 31)) {

#if defined(TARGET_PPC64)

            mb += 32;

            me += 32;

#endif

            tcg_gen_andi_tl(cpu_gpr[rA(ctx->opcode)], t0, MASK(mb, me));

        } else {

            tcg_gen_andi_tl(t0, t0, MASK(32, 63));

            tcg_gen_mov_tl(cpu_gpr[rA(ctx->opcode)], t0);

        }

        tcg_temp_free(t0);

    }

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rA(ctx->opcode)]);

}
