static void gen_jmp_tb(DisasContext *s, target_ulong eip, int tb_num)

{

    gen_update_cc_op(s);

    set_cc_op(s, CC_OP_DYNAMIC);

    if (s->jmp_opt) {

        gen_goto_tb(s, tb_num, eip);

        s->is_jmp = DISAS_TB_JUMP;

    } else {

        gen_jmp_im(eip);

        gen_eob(s);

    }

}
