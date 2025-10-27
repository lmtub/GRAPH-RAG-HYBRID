int qemu_init_main_loop(void)

{

    int ret;



    ret = qemu_signal_init();

    if (ret) {

        return ret;

    }



    qemu_init_sigbus();



    return qemu_event_init();

}
