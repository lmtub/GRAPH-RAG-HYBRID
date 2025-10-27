void op_dmtc0_ebase (void)

{

    /* vectored interrupts not implemented */

    /* Multi-CPU not implemented */

    /* XXX: 64bit addressing broken */

    env->CP0_EBase = (int32_t)0x80000000 | (T0 & 0x3FFFF000);

    RETURN();

}
