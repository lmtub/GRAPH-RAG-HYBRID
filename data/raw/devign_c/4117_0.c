static int gen_jz_ecx_string(DisasContext *s, target_ulong next_eip)

{

    int l1, l2;



    l1 = gen_new_label();

    l2 = gen_new_label();

    gen_op_jnz_ecx[s->aflag](l1);

    gen_set_label(l2);

    gen_jmp_tb(s, next_eip, 1);

    gen_set_label(l1);

    return l2;

}
