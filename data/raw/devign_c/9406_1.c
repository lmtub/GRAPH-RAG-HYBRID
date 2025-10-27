void OPPROTO op_POWER_sllq (void)

{

    uint32_t msk = -1;



    msk = msk << (T1 & 0x1FUL);

    if (T1 & 0x20UL)

        msk = ~msk;

    T1 &= 0x1FUL;

    T0 = (T0 << T1) & msk;

    T0 |= env->spr[SPR_MQ] & ~msk;

    RETURN();

}
