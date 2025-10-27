static int xen_9pfs_connect(struct XenDevice *xendev)

{

    int i;

    Xen9pfsDev *xen_9pdev = container_of(xendev, Xen9pfsDev, xendev);

    V9fsState *s = &xen_9pdev->state;

    QemuOpts *fsdev;



    if (xenstore_read_fe_int(&xen_9pdev->xendev, "num-rings",

                             &xen_9pdev->num_rings) == -1 ||

        xen_9pdev->num_rings > MAX_RINGS || xen_9pdev->num_rings < 1) {

        return -1;

    }



    xen_9pdev->rings = g_malloc0(xen_9pdev->num_rings * sizeof(Xen9pfsRing));

    for (i = 0; i < xen_9pdev->num_rings; i++) {

        char *str;

        int ring_order;



        xen_9pdev->rings[i].priv = xen_9pdev;

        xen_9pdev->rings[i].evtchn = -1;

        xen_9pdev->rings[i].local_port = -1;



        str = g_strdup_printf("ring-ref%u", i);

        if (xenstore_read_fe_int(&xen_9pdev->xendev, str,

                                 &xen_9pdev->rings[i].ref) == -1) {


            goto out;

        }


        str = g_strdup_printf("event-channel-%u", i);

        if (xenstore_read_fe_int(&xen_9pdev->xendev, str,

                                 &xen_9pdev->rings[i].evtchn) == -1) {


            goto out;

        }




        xen_9pdev->rings[i].intf =  xengnttab_map_grant_ref(

                xen_9pdev->xendev.gnttabdev,

                xen_9pdev->xendev.dom,

                xen_9pdev->rings[i].ref,

                PROT_READ | PROT_WRITE);

        if (!xen_9pdev->rings[i].intf) {

            goto out;

        }

        ring_order = xen_9pdev->rings[i].intf->ring_order;

        if (ring_order > MAX_RING_ORDER) {

            goto out;

        }

        xen_9pdev->rings[i].ring_order = ring_order;

        xen_9pdev->rings[i].data = xengnttab_map_domain_grant_refs(

                xen_9pdev->xendev.gnttabdev,

                (1 << ring_order),

                xen_9pdev->xendev.dom,

                xen_9pdev->rings[i].intf->ref,

                PROT_READ | PROT_WRITE);

        if (!xen_9pdev->rings[i].data) {

            goto out;

        }

        xen_9pdev->rings[i].ring.in = xen_9pdev->rings[i].data;

        xen_9pdev->rings[i].ring.out = xen_9pdev->rings[i].data +

                                       XEN_FLEX_RING_SIZE(ring_order);



        xen_9pdev->rings[i].bh = qemu_bh_new(xen_9pfs_bh, &xen_9pdev->rings[i]);

        xen_9pdev->rings[i].out_cons = 0;

        xen_9pdev->rings[i].out_size = 0;

        xen_9pdev->rings[i].inprogress = false;





        xen_9pdev->rings[i].evtchndev = xenevtchn_open(NULL, 0);

        if (xen_9pdev->rings[i].evtchndev == NULL) {

            goto out;

        }

        fcntl(xenevtchn_fd(xen_9pdev->rings[i].evtchndev), F_SETFD, FD_CLOEXEC);

        xen_9pdev->rings[i].local_port = xenevtchn_bind_interdomain

                                            (xen_9pdev->rings[i].evtchndev,

                                             xendev->dom,

                                             xen_9pdev->rings[i].evtchn);

        if (xen_9pdev->rings[i].local_port == -1) {

            xen_pv_printf(xendev, 0,

                          "xenevtchn_bind_interdomain failed port=%d\n",

                          xen_9pdev->rings[i].evtchn);

            goto out;

        }

        xen_pv_printf(xendev, 2, "bind evtchn port %d\n", xendev->local_port);

        qemu_set_fd_handler(xenevtchn_fd(xen_9pdev->rings[i].evtchndev),

                xen_9pfs_evtchn_event, NULL, &xen_9pdev->rings[i]);

    }



    xen_9pdev->security_model = xenstore_read_be_str(xendev, "security_model");

    xen_9pdev->path = xenstore_read_be_str(xendev, "path");

    xen_9pdev->id = s->fsconf.fsdev_id =

        g_strdup_printf("xen9p%d", xendev->dev);

    xen_9pdev->tag = s->fsconf.tag = xenstore_read_fe_str(xendev, "tag");

    v9fs_register_transport(s, &xen_9p_transport);

    fsdev = qemu_opts_create(qemu_find_opts("fsdev"),

            s->fsconf.tag,

            1, NULL);

    qemu_opt_set(fsdev, "fsdriver", "local", NULL);

    qemu_opt_set(fsdev, "path", xen_9pdev->path, NULL);

    qemu_opt_set(fsdev, "security_model", xen_9pdev->security_model, NULL);

    qemu_opts_set_id(fsdev, s->fsconf.fsdev_id);

    qemu_fsdev_add(fsdev);

    v9fs_device_realize_common(s, NULL);



    return 0;



out:

    xen_9pfs_free(xendev);

    return -1;

}