static int get_S2prot(CPUARMState *env, int s2ap, int xn)

{

    int prot = 0;



    if (s2ap & 1) {

        prot |= PAGE_READ;

    }

    if (s2ap & 2) {

        prot |= PAGE_WRITE;

    }

    if (!xn) {

        prot |= PAGE_EXEC;

    }

    return prot;

}
