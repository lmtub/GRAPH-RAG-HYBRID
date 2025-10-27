static void close_htab_fd(sPAPRMachineState *spapr)

{

    if (spapr->htab_fd >= 0) {

        close(spapr->htab_fd);

    }

    spapr->htab_fd = -1;

}
