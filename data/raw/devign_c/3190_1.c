static gboolean udp_chr_read(GIOChannel *chan, GIOCondition cond, void *opaque)

{

    CharDriverState *chr = opaque;

    NetCharDriver *s = chr->opaque;

    gsize bytes_read = 0;

    GIOStatus status;



    if (s->max_size == 0) {

        return TRUE;

    }

    status = g_io_channel_read_chars(s->chan, (gchar *)s->buf, sizeof(s->buf),

                                     &bytes_read, NULL);

    s->bufcnt = bytes_read;

    s->bufptr = s->bufcnt;

    if (status != G_IO_STATUS_NORMAL) {

        if (s->tag) {

            g_source_remove(s->tag);

            s->tag = 0;

        }

        return FALSE;

    }



    s->bufptr = 0;

    while (s->max_size > 0 && s->bufptr < s->bufcnt) {

        qemu_chr_be_write(chr, &s->buf[s->bufptr], 1);

        s->bufptr++;

        s->max_size = qemu_chr_be_can_write(chr);

    }



    return TRUE;

}
