void OPPROTO op_addw_ESI_T0(void)

{

    ESI = (ESI & ~0xffff) | ((ESI + T0) & 0xffff);

}
