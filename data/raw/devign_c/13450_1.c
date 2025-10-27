void OPPROTO op_POWER_slq (void)

{

    uint32_t msk = -1, tmp;



    msk = msk << (T1 & 0x1FUL);

    if (T1 & 0x20UL)

        msk = ~msk;

    T1 &= 0x1FUL;

    tmp = rotl32(T0, T1);

    T0 = tmp & msk;

    env->spr[SPR_MQ] = tmp;

    RETURN();

}
