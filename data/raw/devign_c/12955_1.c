static int con_initialise(struct XenDevice *xendev)

{

    struct XenConsole *con = container_of(xendev, struct XenConsole, xendev);

    int limit;



    if (xenstore_read_int(con->console, "ring-ref", &con->ring_ref) == -1)

	return -1;

    if (xenstore_read_int(con->console, "port", &con->xendev.remote_port) == -1)

	return -1;

    if (xenstore_read_int(con->console, "limit", &limit) == 0)

	con->buffer.max_capacity = limit;



    if (!xendev->dev) {

        con->sring = xc_map_foreign_range(xen_xc, con->xendev.dom,

                                          XC_PAGE_SIZE,

                                          PROT_READ|PROT_WRITE,

                                          con->ring_ref);

    } else {

        con->sring = xengnttab_map_grant_ref(xendev->gnttabdev, con->xendev.dom,

                                             con->ring_ref,

                                             PROT_READ|PROT_WRITE);

    }

    if (!con->sring)

	return -1;



    xen_be_bind_evtchn(&con->xendev);

    if (con->chr) {

        if (qemu_chr_fe_claim(con->chr) == 0) {

            qemu_chr_add_handlers(con->chr, xencons_can_receive,

                                  xencons_receive, NULL, con);

        } else {

            xen_be_printf(xendev, 0,

                          "xen_console_init error chardev %s already used\n",

                          con->chr->label);

            con->chr = NULL;

        }

    }



    xen_be_printf(xendev, 1, "ring mfn %d, remote port %d, local port %d, limit %zd\n",

		  con->ring_ref,

		  con->xendev.remote_port,

		  con->xendev.local_port,

		  con->buffer.max_capacity);

    return 0;

}
