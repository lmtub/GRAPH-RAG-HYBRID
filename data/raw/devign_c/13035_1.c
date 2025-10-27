static gboolean pty_chr_timer(gpointer opaque)

{

    struct CharDriverState *chr = opaque;

    PtyCharDriver *s = chr->opaque;



    if (s->connected) {

        goto out;

    }



    /* Next poll ... */

    pty_chr_update_read_handler(chr);



out:

    s->timer_tag = 0;

    return FALSE;

}
