void OPPROTO op_lmsw_T0(void)

{

    /* only 4 lower bits of CR0 are modified */

    T0 = (env->cr[0] & ~0xf) | (T0 & 0xf);

    helper_movl_crN_T0(0);

}
