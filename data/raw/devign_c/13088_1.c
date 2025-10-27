static ExitStatus trans_fop_wed_0c(DisasContext *ctx, uint32_t insn,

                                   const DisasInsn *di)

{

    unsigned rt = extract32(insn, 0, 5);

    unsigned ra = extract32(insn, 21, 5);

    return do_fop_wed(ctx, rt, ra, di->f_wed);

}
