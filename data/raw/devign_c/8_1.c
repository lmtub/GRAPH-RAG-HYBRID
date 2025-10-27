static void filter_mirror_setup(NetFilterState *nf, Error **errp)
{
    MirrorState *s = FILTER_MIRROR(nf);
    Chardev *chr;
    chr = qemu_chr_find(s->outdev);
    if (chr == NULL) {
        error_set(errp, ERROR_CLASS_DEVICE_NOT_FOUND,
                  "Device '%s' not found", s->outdev);
    qemu_chr_fe_init(&s->chr_out, chr, errp);