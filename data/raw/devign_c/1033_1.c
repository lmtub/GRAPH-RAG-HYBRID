static always_inline void gen_rldimi (DisasContext *ctx, int mbn, int shn)

{

    uint64_t mask;

    uint32_t sh, mb;



    sh = SH(ctx->opcode) | (shn << 5);

    mb = MB(ctx->opcode) | (mbn << 5);

    if (likely(sh == 0)) {

        if (likely(mb == 0)) {

            gen_op_load_gpr_T0(rS(ctx->opcode));

            goto do_store;

        } else if (likely(mb == 63)) {

            gen_op_load_gpr_T0(rA(ctx->opcode));

            goto do_store;

        }

        gen_op_load_gpr_T0(rS(ctx->opcode));

        gen_op_load_gpr_T1(rA(ctx->opcode));

        goto do_mask;

    }

    gen_op_load_gpr_T0(rS(ctx->opcode));

    gen_op_load_gpr_T1(rA(ctx->opcode));

    gen_op_rotli64_T0(sh);

 do_mask:

    mask = MASK(mb, 63 - sh);

    gen_andi_T0_64(ctx, mask);

    gen_andi_T1_64(ctx, ~mask);

    gen_op_or();

 do_store:

    gen_op_store_T0_gpr(rA(ctx->opcode));

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx);

}
