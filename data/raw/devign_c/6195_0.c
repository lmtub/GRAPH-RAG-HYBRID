void qemu_chr_free(CharDriverState *chr)

{

    if (chr->chr_close) {

        chr->chr_close(chr);

    }

    g_free(chr->filename);

    g_free(chr->label);

    qemu_opts_del(chr->opts);

    g_free(chr);

}
