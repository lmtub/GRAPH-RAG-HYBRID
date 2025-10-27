CharDriverState *qemu_chr_alloc(ChardevCommon *backend, Error **errp)

{

    CharDriverState *chr = g_malloc0(sizeof(CharDriverState));

    qemu_mutex_init(&chr->chr_write_lock);



    chr->mux_idx = -1;

    if (backend->has_logfile) {

        int flags = O_WRONLY | O_CREAT;

        if (backend->has_logappend &&

            backend->logappend) {

            flags |= O_APPEND;

        } else {

            flags |= O_TRUNC;

        }

        chr->logfd = qemu_open(backend->logfile, flags, 0666);

        if (chr->logfd < 0) {

            error_setg_errno(errp, errno,

                             "Unable to open logfile %s",

                             backend->logfile);

            g_free(chr);

            return NULL;

        }

    } else {

        chr->logfd = -1;

    }



    return chr;

}
