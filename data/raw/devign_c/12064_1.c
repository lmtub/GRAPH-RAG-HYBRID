static void spapr_finalize_fdt(sPAPRMachineState *spapr,

                               hwaddr fdt_addr,

                               hwaddr rtas_addr,

                               hwaddr rtas_size)

{

    MachineState *machine = MACHINE(qdev_get_machine());

    sPAPRMachineClass *smc = SPAPR_MACHINE_GET_CLASS(machine);

    const char *boot_device = machine->boot_order;

    int ret, i;

    size_t cb = 0;

    char *bootlist;

    void *fdt;

    sPAPRPHBState *phb;



    fdt = g_malloc(FDT_MAX_SIZE);



    /* open out the base tree into a temp buffer for the final tweaks */

    _FDT((fdt_open_into(spapr->fdt_skel, fdt, FDT_MAX_SIZE)));



    ret = spapr_populate_memory(spapr, fdt);

    if (ret < 0) {

        fprintf(stderr, "couldn't setup memory nodes in fdt\n");

        exit(1);

    }



    ret = spapr_populate_vdevice(spapr->vio_bus, fdt);

    if (ret < 0) {

        fprintf(stderr, "couldn't setup vio devices in fdt\n");

        exit(1);

    }



    if (object_resolve_path_type("", TYPE_SPAPR_RNG, NULL)) {

        ret = spapr_rng_populate_dt(fdt);

        if (ret < 0) {

            fprintf(stderr, "could not set up rng device in the fdt\n");

            exit(1);

        }

    }



    QLIST_FOREACH(phb, &spapr->phbs, list) {

        ret = spapr_populate_pci_dt(phb, PHANDLE_XICP, fdt);

        if (ret < 0) {

            error_report("couldn't setup PCI devices in fdt");

            exit(1);

        }

    }



    /* RTAS */

    ret = spapr_rtas_device_tree_setup(fdt, rtas_addr, rtas_size);

    if (ret < 0) {

        fprintf(stderr, "Couldn't set up RTAS device tree properties\n");

    }



    /* cpus */

    spapr_populate_cpus_dt_node(fdt, spapr);



    bootlist = get_boot_devices_list(&cb, true);

    if (cb && bootlist) {

        int offset = fdt_path_offset(fdt, "/chosen");

        if (offset < 0) {

            exit(1);

        }

        for (i = 0; i < cb; i++) {

            if (bootlist[i] == '\n') {

                bootlist[i] = ' ';

            }



        }

        ret = fdt_setprop_string(fdt, offset, "qemu,boot-list", bootlist);

    }



    if (boot_device && strlen(boot_device)) {

        int offset = fdt_path_offset(fdt, "/chosen");



        if (offset < 0) {

            exit(1);

        }

        fdt_setprop_string(fdt, offset, "qemu,boot-device", boot_device);

    }



    if (!spapr->has_graphics) {

        spapr_populate_chosen_stdout(fdt, spapr->vio_bus);

    }



    if (smc->dr_lmb_enabled) {

        _FDT(spapr_drc_populate_dt(fdt, 0, NULL, SPAPR_DR_CONNECTOR_TYPE_LMB));

    }



    if (smc->dr_cpu_enabled) {

        int offset = fdt_path_offset(fdt, "/cpus");

        ret = spapr_drc_populate_dt(fdt, offset, NULL,

                                    SPAPR_DR_CONNECTOR_TYPE_CPU);

        if (ret < 0) {

            error_report("Couldn't set up CPU DR device tree properties");

            exit(1);

        }

    }



    _FDT((fdt_pack(fdt)));



    if (fdt_totalsize(fdt) > FDT_MAX_SIZE) {

        error_report("FDT too big ! 0x%x bytes (max is 0x%x)",

                     fdt_totalsize(fdt), FDT_MAX_SIZE);

        exit(1);

    }



    qemu_fdt_dumpdtb(fdt, fdt_totalsize(fdt));

    cpu_physical_memory_write(fdt_addr, fdt, fdt_totalsize(fdt));



    g_free(bootlist);

    g_free(fdt);

}
