static void xen_config_cleanup_dir(char *dir)

{

    struct xs_dirs *d;



    d = qemu_malloc(sizeof(*d));

    d->xs_dir = dir;

    TAILQ_INSERT_TAIL(&xs_cleanup, d, list);

}
