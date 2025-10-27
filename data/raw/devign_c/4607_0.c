static CharDriverState *qemu_chr_open_null(const char *id,

                                           ChardevBackend *backend,

                                           ChardevReturn *ret,

                                           Error **errp)

{

    CharDriverState *chr;

    ChardevCommon *common = backend->u.null;



    chr = qemu_chr_alloc(common, errp);

    if (!chr) {

        return NULL;

    }

    chr->chr_write = null_chr_write;

    chr->explicit_be_open = true;

    return chr;

}
