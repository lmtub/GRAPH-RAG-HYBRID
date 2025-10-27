static void hash32_bat_size(CPUPPCState *env, target_ulong *blp, int *validp,

                            target_ulong batu, target_ulong batl)

{

    target_ulong bl;

    int valid;



    bl = (batu & BATU32_BL) << 15;

    valid = 0;

    if (((msr_pr == 0) && (batu & BATU32_VS)) ||

        ((msr_pr != 0) && (batu & BATU32_VP))) {

        valid = 1;

    }

    *blp = bl;

    *validp = valid;

}
