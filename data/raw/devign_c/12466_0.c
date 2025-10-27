int xen_domain_build_pv(const char *kernel, const char *ramdisk,

                        const char *cmdline)

{

    uint32_t ssidref = 0;

    uint32_t flags = 0;

    xen_domain_handle_t uuid;

    unsigned int xenstore_port = 0, console_port = 0;

    unsigned long xenstore_mfn = 0, console_mfn = 0;

    int rc;



    memcpy(uuid, qemu_uuid, sizeof(uuid));

    rc = xc_domain_create(xen_xc, ssidref, uuid, flags, &xen_domid);

    if (rc < 0) {

        fprintf(stderr, "xen: xc_domain_create() failed\n");

        goto err;

    }

    qemu_log("xen: created domain %d\n", xen_domid);

    atexit(xen_domain_cleanup);

    xen_domain_watcher();



    xenstore_domain_init1(kernel, ramdisk, cmdline);



    rc = xc_domain_max_vcpus(xen_xc, xen_domid, smp_cpus);

    if (rc < 0) {

        fprintf(stderr, "xen: xc_domain_max_vcpus() failed\n");

        goto err;

    }



#if 0

    rc = xc_domain_setcpuweight(xen_xc, xen_domid, 256);

    if (rc < 0) {

        fprintf(stderr, "xen: xc_domain_setcpuweight() failed\n");

        goto err;

    }

#endif



    rc = xc_domain_setmaxmem(xen_xc, xen_domid, ram_size >> 10);

    if (rc < 0) {

        fprintf(stderr, "xen: xc_domain_setmaxmem() failed\n");

        goto err;

    }



    xenstore_port = xc_evtchn_alloc_unbound(xen_xc, xen_domid, 0);

    console_port = xc_evtchn_alloc_unbound(xen_xc, xen_domid, 0);



    rc = xc_linux_build(xen_xc, xen_domid, ram_size >> 20,

                        kernel, ramdisk, cmdline,

                        0, flags,

                        xenstore_port, &xenstore_mfn,

                        console_port, &console_mfn);

    if (rc < 0) {

        fprintf(stderr, "xen: xc_linux_build() failed\n");

        goto err;

    }



    xenstore_domain_init2(xenstore_port, xenstore_mfn,

                          console_port, console_mfn);



    qemu_log("xen: unpausing domain %d\n", xen_domid);

    rc = xc_domain_unpause(xen_xc, xen_domid);

    if (rc < 0) {

        fprintf(stderr, "xen: xc_domain_unpause() failed\n");

        goto err;

    }



    xen_poll = qemu_new_timer(rt_clock, xen_domain_poll, NULL);

    qemu_mod_timer(xen_poll, qemu_get_clock(rt_clock) + 1000);

    return 0;



err:

    return -1;

}
