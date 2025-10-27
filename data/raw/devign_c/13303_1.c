static void spapr_finalize_fdt(sPAPREnvironment *spapr,

                               hwaddr fdt_addr,

                               hwaddr rtas_addr,

                               hwaddr rtas_size)

{

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



    QLIST_FOREACH(phb, &spapr->phbs, list) {

        ret = spapr_populate_pci_dt(phb, PHANDLE_XICP, fdt);

    }



    if (ret < 0) {

        fprintf(stderr, "couldn't setup PCI devices in fdt\n");

        exit(1);

    }



    /* RTAS */

    ret = spapr_rtas_device_tree_setup(fdt, rtas_addr, rtas_size);

    if (ret < 0) {

        fprintf(stderr, "Couldn't set up RTAS device tree properties\n");

    }



    /* Advertise NUMA via ibm,associativity */

    ret = spapr_fixup_cpu_dt(fdt, spapr);

    if (ret < 0) {

        fprintf(stderr, "Couldn't finalize CPU device tree properties\n");

    }



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



    if (!spapr->has_graphics) {

        spapr_populate_chosen_stdout(fdt, spapr->vio_bus);

    }



    _FDT((fdt_pack(fdt)));



    if (fdt_totalsize(fdt) > FDT_MAX_SIZE) {

        hw_error("FDT too big ! 0x%x bytes (max is 0x%x)\n",

                 fdt_totalsize(fdt), FDT_MAX_SIZE);

        exit(1);

    }



    cpu_physical_memory_write(fdt_addr, fdt, fdt_totalsize(fdt));




    g_free(fdt);

}