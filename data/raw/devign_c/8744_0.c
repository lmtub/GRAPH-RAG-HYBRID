static inline void gen_scas(DisasContext *s, int ot)

{

    gen_op_mov_TN_reg(OT_LONG, 0, R_EAX);

    gen_string_movl_A0_EDI(s);

    gen_op_ld_T1_A0(ot + s->mem_index);

    gen_op_cmpl_T0_T1_cc();

    gen_op_movl_T0_Dshift[ot]();

#ifdef TARGET_X86_64

    if (s->aflag == 2) {

        gen_op_addq_EDI_T0();

    } else

#endif

    if (s->aflag) {

        gen_op_addl_EDI_T0();

    } else {

        gen_op_addw_EDI_T0();

    }

}
