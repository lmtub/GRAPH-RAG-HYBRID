static void qemu_chr_parse_ringbuf(QemuOpts *opts, ChardevBackend *backend,

                                   Error **errp)

{

    int val;

    ChardevRingbuf *ringbuf;



    ringbuf = backend->u.ringbuf = g_new0(ChardevRingbuf, 1);

    qemu_chr_parse_common(opts, qapi_ChardevRingbuf_base(ringbuf));



    val = qemu_opt_get_size(opts, "size", 0);

    if (val != 0) {

        ringbuf->has_size = true;

        ringbuf->size = val;

    }

}
