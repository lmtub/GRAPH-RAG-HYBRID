static ExitStatus trans_fop_dew_0e(DisasContext *ctx, uint32_t insn,

                                   const DisasInsn *di)

{

    unsigned rt = extract32(insn, 0, 5);

    unsigned ra = assemble_ra64(insn);

    return do_fop_dew(ctx, rt, ra, di->f_dew);

}
