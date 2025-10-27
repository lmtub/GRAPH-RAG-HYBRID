static int check_pow_970 (CPUPPCState *env)

{

    if (env->spr[SPR_HID0] & 0x00600000)

        return 1;



    return 0;

}
