void OPPROTO op_fdivr_STN_ST0(void)

{

    CPU86_LDouble *p;

    p = &ST(PARAM1);

    *p = ST0 / *p;

}
