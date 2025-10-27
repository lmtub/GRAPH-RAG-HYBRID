void do_POWER_maskg (void)

{

    uint32_t ret;



    if ((uint32_t)T0 == (uint32_t)(T1 + 1)) {

        ret = -1;

    } else {

        ret = (((uint32_t)(-1)) >> ((uint32_t)T0)) ^

            (((uint32_t)(-1) >> ((uint32_t)T1)) >> 1);

        if ((uint32_t)T0 > (uint32_t)T1)

            ret = ~ret;

    }

    T0 = ret;

}
