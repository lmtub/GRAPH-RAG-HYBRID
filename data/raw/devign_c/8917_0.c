void OPPROTO op_jz_T0_label(void)

{

    if (!T0)

        GOTO_LABEL_PARAM(1);

    FORCE_RET();

}
