void OPPROTO op_int_im(void)

{

    EIP = PARAM1;

    raise_exception(EXCP0D_GPF);

}
