static void xen_domain_watcher(void)

{

    int qemu_running = 1;

    int fd[2], i, n, rc;

    char byte;



    pipe(fd);

    if (fork() != 0)

        return; /* not child */



    /* close all file handles, except stdio/out/err,

     * our watch pipe and the xen interface handle */

    n = getdtablesize();

    for (i = 3; i < n; i++) {

        if (i == fd[0])

            continue;

        if (i == xen_xc)

            continue;

        close(i);

    }



    /* ignore term signals */

    signal(SIGINT,  SIG_IGN);

    signal(SIGTERM, SIG_IGN);



    /* wait for qemu exiting */

    while (qemu_running) {

        rc = read(fd[0], &byte, 1);

        switch (rc) {

        case -1:

            if (errno == EINTR)

                continue;

            qemu_log("%s: Huh? read error: %s\n", __FUNCTION__, strerror(errno));

            qemu_running = 0;

            break;

        case 0:

            /* EOF -> qemu exited */

            qemu_running = 0;

            break;

        default:

            qemu_log("%s: Huh? data on the watch pipe?\n", __FUNCTION__);

            break;

        }

    }



    /* cleanup */

    qemu_log("%s: destroy domain %d\n", __FUNCTION__, xen_domid);

    xc_domain_destroy(xen_xc, xen_domid);

    _exit(0);

}
