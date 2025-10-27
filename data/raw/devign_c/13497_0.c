void qemu_main_loop_start(void)

{

    qemu_system_ready = 1;

    qemu_cond_broadcast(&qemu_system_cond);

}
