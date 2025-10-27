void OPPROTO op_addw_EDI_T0(void)

{

    EDI = (EDI & ~0xffff) | ((EDI + T0) & 0xffff);

}
