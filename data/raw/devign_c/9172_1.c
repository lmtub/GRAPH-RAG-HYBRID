VIOsPAPRDevice *vty_lookup(sPAPRMachineState *spapr, target_ulong reg)
{
    VIOsPAPRDevice *sdev;
    sdev = spapr_vio_find_by_reg(spapr->vio_bus, reg);
    if (!sdev && reg == 0) {
        /* Hack for kernel early debug, which always specifies reg==0.
         * We search all VIO devices, and grab the vty with the lowest
         * reg.  This attempts to mimic existing PowerVM behaviour
         * (early debug does work there, despite having no vty with
         * reg==0. */
        return spapr_vty_get_default(spapr->vio_bus);
    return sdev;