void xen_config_cleanup(void)

{

    struct xs_dirs *d;



    TAILQ_FOREACH(d, &xs_cleanup, list) {

	xs_rm(xenstore, 0, d->xs_dir);

    }

}
