static gboolean qemu_chr_be_generic_open_bh(gpointer opaque)

{

    CharDriverState *s = opaque;

    qemu_chr_be_event(s, CHR_EVENT_OPENED);

    s->idle_tag = 0;

    return FALSE;

}
