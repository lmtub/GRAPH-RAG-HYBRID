void monitor_flush(Monitor *mon)

{

    int rc;

    size_t len;

    const char *buf;



    if (mon->skip_flush) {

        return;

    }



    buf = qstring_get_str(mon->outbuf);

    len = qstring_get_length(mon->outbuf);



    if (len && !mon->mux_out) {

        rc = qemu_chr_fe_write(mon->chr, (const uint8_t *) buf, len);

        if (rc == len) {

            /* all flushed */

            QDECREF(mon->outbuf);

            mon->outbuf = qstring_new();

            return;

        }

        if (rc > 0) {

            /* partinal write */

            QString *tmp = qstring_from_str(buf + rc);

            QDECREF(mon->outbuf);

            mon->outbuf = tmp;

        }

        if (mon->watch == 0) {

            mon->watch = qemu_chr_fe_add_watch(mon->chr, G_IO_OUT,

                                               monitor_unblocked, mon);

        }

    }

}
