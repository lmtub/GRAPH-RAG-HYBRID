static void spapr_alloc_htab(sPAPRMachineState *spapr)

{

    long shift;

    int index;



    /* allocate hash page table.  For now we always make this 16mb,

     * later we should probably make it scale to the size of guest

     * RAM */



    shift = kvmppc_reset_htab(spapr->htab_shift);



    if (shift > 0) {

        /* Kernel handles htab, we don't need to allocate one */

        if (shift != spapr->htab_shift) {

            error_setg(&error_abort, "Failed to allocate HTAB of requested size, try with smaller maxmem");

        }



        spapr->htab_shift = shift;

        kvmppc_kern_htab = true;

    } else {

        /* Allocate htab */

        spapr->htab = qemu_memalign(HTAB_SIZE(spapr), HTAB_SIZE(spapr));



        /* And clear it */

        memset(spapr->htab, 0, HTAB_SIZE(spapr));



        for (index = 0; index < HTAB_SIZE(spapr) / HASH_PTE_SIZE_64; index++) {

            DIRTY_HPTE(HPTE(spapr->htab, index));

        }

    }

}
