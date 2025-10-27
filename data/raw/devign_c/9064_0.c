DISAS_INSN(scc)

{

    int l1;

    int cond;

    TCGv reg;



    l1 = gen_new_label();

    cond = (insn >> 8) & 0xf;

    reg = DREG(insn, 0);

    tcg_gen_andi_i32(reg, reg, 0xffffff00);

    /* This is safe because we modify the reg directly, with no other values

       live.  */

    gen_jmpcc(s, cond ^ 1, l1);

    tcg_gen_ori_i32(reg, reg, 0xff);

    gen_set_label(l1);

}
