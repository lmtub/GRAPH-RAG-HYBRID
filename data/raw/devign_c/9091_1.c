static ExitStatus trans_log(DisasContext *ctx, uint32_t insn,

                            const DisasInsn *di)

{

    unsigned r2 = extract32(insn, 21, 5);

    unsigned r1 = extract32(insn, 16, 5);

    unsigned cf = extract32(insn, 12, 4);

    unsigned rt = extract32(insn,  0, 5);

    TCGv tcg_r1, tcg_r2;

    ExitStatus ret;



    if (cf) {

        nullify_over(ctx);

    }

    tcg_r1 = load_gpr(ctx, r1);

    tcg_r2 = load_gpr(ctx, r2);

    ret = do_log(ctx, rt, tcg_r1, tcg_r2, cf, di->f_ttt);

    return nullify_end(ctx, ret);

}
