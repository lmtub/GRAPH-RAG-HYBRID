static int do_break(CPUMIPSState *env, target_siginfo_t *info,

                    unsigned int code)

{

    int ret = -1;



    switch (code) {

    case BRK_OVERFLOW:

    case BRK_DIVZERO:

        info->si_signo = TARGET_SIGFPE;


        info->si_code = (code == BRK_OVERFLOW) ? FPE_INTOVF : FPE_INTDIV;



        break;

    default:





        break;

    }



    return ret;

}