void xen_hvm_init(PCMachineState *pcms, MemoryRegion **ram_memory)

{

    int i, rc;

    xen_pfn_t ioreq_pfn;

    xen_pfn_t bufioreq_pfn;

    evtchn_port_t bufioreq_evtchn;

    XenIOState *state;



    state = g_malloc0(sizeof (XenIOState));



    state->xce_handle = xen_xc_evtchn_open(NULL, 0);

    if (state->xce_handle == XC_HANDLER_INITIAL_VALUE) {

        perror("xen: event channel open");

        goto err;

    }



    state->xenstore = xs_daemon_open();

    if (state->xenstore == NULL) {

        perror("xen: xenstore open");

        goto err;

    }



    rc = xen_create_ioreq_server(xen_xc, xen_domid, &state->ioservid);

    if (rc < 0) {

        perror("xen: ioreq server create");

        goto err;

    }



    state->exit.notify = xen_exit_notifier;

    qemu_add_exit_notifier(&state->exit);



    state->suspend.notify = xen_suspend_notifier;

    qemu_register_suspend_notifier(&state->suspend);



    state->wakeup.notify = xen_wakeup_notifier;

    qemu_register_wakeup_notifier(&state->wakeup);



    rc = xen_get_ioreq_server_info(xen_xc, xen_domid, state->ioservid,

                                   &ioreq_pfn, &bufioreq_pfn,

                                   &bufioreq_evtchn);

    if (rc < 0) {

        error_report("failed to get ioreq server info: error %d handle=" XC_INTERFACE_FMT,

                     errno, xen_xc);

        goto err;

    }



    DPRINTF("shared page at pfn %lx\n", ioreq_pfn);

    DPRINTF("buffered io page at pfn %lx\n", bufioreq_pfn);

    DPRINTF("buffered io evtchn is %x\n", bufioreq_evtchn);



    state->shared_page = xc_map_foreign_range(xen_xc, xen_domid, XC_PAGE_SIZE,

                                              PROT_READ|PROT_WRITE, ioreq_pfn);

    if (state->shared_page == NULL) {

        error_report("map shared IO page returned error %d handle=" XC_INTERFACE_FMT,

                     errno, xen_xc);

        goto err;

    }



    rc = xen_get_vmport_regs_pfn(xen_xc, xen_domid, &ioreq_pfn);

    if (!rc) {

        DPRINTF("shared vmport page at pfn %lx\n", ioreq_pfn);

        state->shared_vmport_page =

            xc_map_foreign_range(xen_xc, xen_domid, XC_PAGE_SIZE,

                                 PROT_READ|PROT_WRITE, ioreq_pfn);

        if (state->shared_vmport_page == NULL) {

            error_report("map shared vmport IO page returned error %d handle="

                         XC_INTERFACE_FMT, errno, xen_xc);

            goto err;

        }

    } else if (rc != -ENOSYS) {

        error_report("get vmport regs pfn returned error %d, rc=%d",

                     errno, rc);

        goto err;

    }



    state->buffered_io_page = xc_map_foreign_range(xen_xc, xen_domid,

                                                   XC_PAGE_SIZE,

                                                   PROT_READ|PROT_WRITE,

                                                   bufioreq_pfn);

    if (state->buffered_io_page == NULL) {

        error_report("map buffered IO page returned error %d", errno);

        goto err;

    }



    /* Note: cpus is empty at this point in init */

    state->cpu_by_vcpu_id = g_malloc0(max_cpus * sizeof(CPUState *));



    rc = xen_set_ioreq_server_state(xen_xc, xen_domid, state->ioservid, true);

    if (rc < 0) {

        error_report("failed to enable ioreq server info: error %d handle=" XC_INTERFACE_FMT,

                     errno, xen_xc);

        goto err;

    }



    state->ioreq_local_port = g_malloc0(max_cpus * sizeof (evtchn_port_t));



    /* FIXME: how about if we overflow the page here? */

    for (i = 0; i < max_cpus; i++) {

        rc = xc_evtchn_bind_interdomain(state->xce_handle, xen_domid,

                                        xen_vcpu_eport(state->shared_page, i));

        if (rc == -1) {

            error_report("shared evtchn %d bind error %d", i, errno);

            goto err;

        }

        state->ioreq_local_port[i] = rc;

    }



    rc = xc_evtchn_bind_interdomain(state->xce_handle, xen_domid,

                                    bufioreq_evtchn);

    if (rc == -1) {

        error_report("buffered evtchn bind error %d", errno);

        goto err;

    }

    state->bufioreq_local_port = rc;



    /* Init RAM management */

    xen_map_cache_init(xen_phys_offset_to_gaddr, state);

    xen_ram_init(pcms, ram_size, ram_memory);



    qemu_add_vm_change_state_handler(xen_hvm_change_state_handler, state);



    state->memory_listener = xen_memory_listener;

    QLIST_INIT(&state->physmap);

    memory_listener_register(&state->memory_listener, &address_space_memory);

    state->log_for_dirtybit = NULL;



    state->io_listener = xen_io_listener;

    memory_listener_register(&state->io_listener, &address_space_io);



    state->device_listener = xen_device_listener;

    device_listener_register(&state->device_listener);



    /* Initialize backend core & drivers */

    if (xen_be_init() != 0) {

        error_report("xen backend core setup failed");

        goto err;

    }

    xen_be_register("console", &xen_console_ops);

    xen_be_register("vkbd", &xen_kbdmouse_ops);

    xen_be_register("qdisk", &xen_blkdev_ops);

    xen_read_physmap(state);

    return;



err:

    error_report("xen hardware virtual machine initialisation failed");

    exit(1);

}
