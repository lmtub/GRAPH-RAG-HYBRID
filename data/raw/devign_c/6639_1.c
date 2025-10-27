static gboolean guest_exec_output_watch(GIOChannel *ch,

        GIOCondition cond, gpointer p_)

{

    GuestExecIOData *p = (GuestExecIOData *)p_;

    gsize bytes_read;

    GIOStatus gstatus;



    if (cond == G_IO_HUP || cond == G_IO_ERR) {

        goto close;

    }



    if (p->size == p->length) {

        gpointer t = NULL;

        if (!p->truncated && p->size < GUEST_EXEC_MAX_OUTPUT) {

            t = g_try_realloc(p->data, p->size + GUEST_EXEC_IO_SIZE);

        }

        if (t == NULL) {

            /* ignore truncated output */

            gchar buf[GUEST_EXEC_IO_SIZE];



            p->truncated = true;

            gstatus = g_io_channel_read_chars(ch, buf, sizeof(buf),

                                              &bytes_read, NULL);

            if (gstatus == G_IO_STATUS_EOF || gstatus == G_IO_STATUS_ERROR) {

                goto close;

            }



            return true;

        }

        p->size += GUEST_EXEC_IO_SIZE;

        p->data = t;

    }



    /* Calling read API once.

     * On next available data our callback will be called again */

    gstatus = g_io_channel_read_chars(ch, (gchar *)p->data + p->length,

            p->size - p->length, &bytes_read, NULL);

    if (gstatus == G_IO_STATUS_EOF || gstatus == G_IO_STATUS_ERROR) {

        goto close;

    }



    p->length += bytes_read;



    return true;



close:


    g_io_channel_unref(ch);

    g_atomic_int_set(&p->closed, 1);

    return false;

}