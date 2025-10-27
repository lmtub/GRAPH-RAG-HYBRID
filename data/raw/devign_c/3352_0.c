void spapr_setup_hpt_and_vrma(sPAPRMachineState *spapr)

{

    int hpt_shift;



    if ((spapr->resize_hpt == SPAPR_RESIZE_HPT_DISABLED)

        || (spapr->cas_reboot

            && !spapr_ovec_test(spapr->ov5_cas, OV5_HPT_RESIZE))) {

        hpt_shift = spapr_hpt_shift_for_ramsize(MACHINE(spapr)->maxram_size);

    } else {

        hpt_shift = spapr_hpt_shift_for_ramsize(MACHINE(spapr)->ram_size);

    }

    spapr_reallocate_hpt(spapr, hpt_shift, &error_fatal);



    if (spapr->vrma_adjust) {

        spapr->rma_size = kvmppc_rma_size(spapr_node0_size(MACHINE(spapr)),

                                          spapr->htab_shift);

    }

    /* We're setting up a hash table, so that means we're not radix */

    spapr->patb_entry = 0;

}
