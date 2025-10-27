static void fd_coroutine_enter(void *opaque)

{

    FDYieldUntilData *data = opaque;

    qemu_set_fd_handler(data->fd, NULL, NULL, NULL);

    qemu_coroutine_enter(data->co, NULL);

}
