void OPPROTO op_mov_T0_cc(void)

{

    T0 = cc_table[CC_OP].compute_all();

}
