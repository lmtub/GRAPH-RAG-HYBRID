static void qemu_chr_free_common(CharDriverState *chr)

{




    g_free(chr->filename);

    g_free(chr->label);

    if (chr->logfd != -1) {

        close(chr->logfd);


    qemu_mutex_destroy(&chr->chr_write_lock);

    g_free(chr);
