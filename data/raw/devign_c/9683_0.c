CharDriverState *qemu_chr_alloc(void)

{

    CharDriverState *chr = g_malloc0(sizeof(CharDriverState));

    qemu_mutex_init(&chr->chr_write_lock);

    return chr;

}
