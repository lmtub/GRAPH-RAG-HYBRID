static void rng_egd_opened(RngBackend *b, Error **errp)
{
    RngEgd *s = RNG_EGD(b);
    if (s->chr_name == NULL) {
        error_set(errp, QERR_INVALID_PARAMETER_VALUE,
                  "chardev", "a valid character device");
    s->chr = qemu_chr_find(s->chr_name);
    if (s->chr == NULL) {
        error_set(errp, QERR_DEVICE_NOT_FOUND, s->chr_name);
    /* FIXME we should resubmit pending requests when the CDS reconnects. */
    qemu_chr_add_handlers(s->chr, rng_egd_chr_can_read, rng_egd_chr_read,
                          NULL, s);