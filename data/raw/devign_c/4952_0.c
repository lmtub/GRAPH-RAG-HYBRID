static void ppc_spapr_reset(void)

{

    sPAPRMachineState *spapr = SPAPR_MACHINE(qdev_get_machine());

    PowerPCCPU *first_ppc_cpu;

    uint32_t rtas_limit;



    /* Check for unknown sysbus devices */

    foreach_dynamic_sysbus_device(find_unknown_sysbus_device, NULL);



    /* Reset the hash table & recalc the RMA */

    spapr_reset_htab(spapr);



    qemu_devices_reset();



    /*

     * We place the device tree and RTAS just below either the top of the RMA,

     * or just below 2GB, whichever is lowere, so that it can be

     * processed with 32-bit real mode code if necessary

     */

    rtas_limit = MIN(spapr->rma_size, RTAS_MAX_ADDR);

    spapr->rtas_addr = rtas_limit - RTAS_MAX_SIZE;

    spapr->fdt_addr = spapr->rtas_addr - FDT_MAX_SIZE;



    /* Load the fdt */

    spapr_finalize_fdt(spapr, spapr->fdt_addr, spapr->rtas_addr,

                       spapr->rtas_size);



    /* Copy RTAS over */

    cpu_physical_memory_write(spapr->rtas_addr, spapr->rtas_blob,

                              spapr->rtas_size);



    /* Set up the entry state */

    first_ppc_cpu = POWERPC_CPU(first_cpu);

    first_ppc_cpu->env.gpr[3] = spapr->fdt_addr;

    first_ppc_cpu->env.gpr[5] = 0;

    first_cpu->halted = 0;

    first_ppc_cpu->env.nip = spapr->entry_point;



}
