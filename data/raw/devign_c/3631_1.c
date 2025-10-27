static void xen_init_pv(QEMUMachineInitArgs *args)

{

    const char *cpu_model = args->cpu_model;

    const char *kernel_filename = args->kernel_filename;

    const char *kernel_cmdline = args->kernel_cmdline;

    const char *initrd_filename = args->initrd_filename;

    X86CPU *cpu;

    CPUState *cs;

    DriveInfo *dinfo;

    int i;



    /* Initialize a dummy CPU */

    if (cpu_model == NULL) {

#ifdef TARGET_X86_64

        cpu_model = "qemu64";

#else

        cpu_model = "qemu32";

#endif

    }

    cpu = cpu_x86_init(cpu_model);

    cs = CPU(cpu);

    cs->halted = 1;



    /* Initialize backend core & drivers */

    if (xen_be_init() != 0) {

        fprintf(stderr, "%s: xen backend core setup failed\n", __FUNCTION__);

        exit(1);

    }



    switch (xen_mode) {

    case XEN_ATTACH:

        /* nothing to do, xend handles everything */

        break;

    case XEN_CREATE:

        if (xen_domain_build_pv(kernel_filename, initrd_filename,

                                kernel_cmdline) < 0) {

            fprintf(stderr, "xen pv domain creation failed\n");

            exit(1);

        }

        break;

    case XEN_EMULATE:

        fprintf(stderr, "xen emulation not implemented (yet)\n");

        exit(1);

        break;

    }



    xen_be_register("console", &xen_console_ops);

    xen_be_register("vkbd", &xen_kbdmouse_ops);

    xen_be_register("vfb", &xen_framebuffer_ops);

    xen_be_register("qdisk", &xen_blkdev_ops);

    xen_be_register("qnic", &xen_netdev_ops);



    /* configure framebuffer */

    if (xenfb_enabled) {

        xen_config_dev_vfb(0, "vnc");

        xen_config_dev_vkbd(0);

    }



    /* configure disks */

    for (i = 0; i < 16; i++) {

        dinfo = drive_get(IF_XEN, 0, i);

        if (!dinfo)

            continue;

        xen_config_dev_blk(dinfo);

    }



    /* configure nics */

    for (i = 0; i < nb_nics; i++) {

        if (!nd_table[i].model || 0 != strcmp(nd_table[i].model, "xen"))

            continue;

        xen_config_dev_nic(nd_table + i);

    }



    /* config cleanup hook */

    atexit(xen_config_cleanup);



    /* setup framebuffer */

    xen_init_display(xen_domid);

}
