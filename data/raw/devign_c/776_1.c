static void gen_ori(DisasContext *ctx)

{

    target_ulong uimm = UIMM(ctx->opcode);



    if (rS(ctx->opcode) == rA(ctx->opcode) && uimm == 0) {

        /* NOP */

        /* XXX: should handle special NOPs for POWER series */

        return;

    }

    tcg_gen_ori_tl(cpu_gpr[rA(ctx->opcode)], cpu_gpr[rS(ctx->opcode)], uimm);

}
