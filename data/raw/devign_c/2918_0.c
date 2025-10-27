static int check_pow_970MP (CPUPPCState *env)

{

    if (env->spr[SPR_HID0] & 0x01C00000)

        return 1;



    return 0;

}
