static void qemu_aio_wait_nonblocking(void)

{

    qemu_notify_event();

    qemu_aio_wait();

}
