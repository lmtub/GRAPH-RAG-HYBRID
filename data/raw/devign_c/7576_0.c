static void xen_domain_poll(void *opaque)

{

    struct xc_dominfo info;

    int rc;



    rc = xc_domain_getinfo(xen_xc, xen_domid, 1, &info);

    if ((rc != 1) || (info.domid != xen_domid)) {

        qemu_log("xen: domain %d is gone\n", xen_domid);

        goto quit;

    }

    if (info.dying) {

        qemu_log("xen: domain %d is dying (%s%s)\n", xen_domid,

                 info.crashed  ? "crashed"  : "",

                 info.shutdown ? "shutdown" : "");

        goto quit;

    }



    qemu_mod_timer(xen_poll, qemu_get_clock(rt_clock) + 1000);

    return;



quit:

    qemu_system_shutdown_request();

    return;

}
