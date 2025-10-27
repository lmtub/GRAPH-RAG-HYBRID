static void dec_bcc(DisasContext *dc)

{

    unsigned int cc;

    unsigned int dslot;



    cc = EXTRACT_FIELD(dc->ir, 21, 23);

    dslot = dc->ir & (1 << 25);

    LOG_DIS("bcc%s r%d %x\n", dslot ? "d" : "", dc->ra, dc->imm);



    dc->delayed_branch = 1;

    if (dslot) {

        dc->delayed_branch = 2;

        dc->tb_flags |= D_FLAG;

        tcg_gen_st_tl(tcg_const_tl(dc->type_b && (dc->tb_flags & IMM_FLAG)),

                      cpu_env, offsetof(CPUState, bimm));

    }



    if (dec_alu_op_b_is_small_imm(dc)) {

        int32_t offset = (int32_t)((int16_t)dc->imm); /* sign-extend.  */



        tcg_gen_movi_tl(env_btarget, dc->pc + offset);

    } else {

        tcg_gen_movi_tl(env_btarget, dc->pc);

        tcg_gen_add_tl(env_btarget, env_btarget, *(dec_alu_op_b(dc)));

    }

    dc->jmp = JMP_INDIRECT;

    eval_cc(dc, cc, env_btaken, cpu_R[dc->ra], tcg_const_tl(0));

}
