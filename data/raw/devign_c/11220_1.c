void OPPROTO op_4xx_tlbsx_check (void)

{

    int tmp;



    tmp = xer_so;

    if (T0 != -1)

        tmp |= 0x02;

    env->crf[0] = tmp;

    RETURN();

}
