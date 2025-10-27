static int xenstore_scan(const char *type, int dom, struct XenDevOps *ops)

{

    struct XenDevice *xendev;

    char path[XEN_BUFSIZE], token[XEN_BUFSIZE];

    char **dev = NULL, *dom0;

    unsigned int cdev, j;



    /* setup watch */

    dom0 = xs_get_domain_path(xenstore, 0);

    snprintf(token, sizeof(token), "be:%p:%d:%p", type, dom, ops);

    snprintf(path, sizeof(path), "%s/backend/%s/%d", dom0, type, dom);

    free(dom0);

    if (!xs_watch(xenstore, path, token)) {

        xen_be_printf(NULL, 0, "xen be: watching backend path (%s) failed\n", path);

        return -1;

    }



    /* look for backends */

    dev = xs_directory(xenstore, 0, path, &cdev);

    if (!dev) {

        return 0;

    }

    for (j = 0; j < cdev; j++) {

        xendev = xen_be_get_xendev(type, dom, atoi(dev[j]), ops);

        if (xendev == NULL) {

            continue;

        }

        xen_be_check_state(xendev);

    }

    free(dev);

    return 0;

}
