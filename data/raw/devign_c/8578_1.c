static ExitStatus trans_fop_wed_0e(DisasContext *ctx, uint32_t insn,

                                   const DisasInsn *di)

{

    unsigned rt = assemble_rt64(insn);

    unsigned ra = extract32(insn, 21, 5);

    return do_fop_wed(ctx, rt, ra, di->f_wed);

}
