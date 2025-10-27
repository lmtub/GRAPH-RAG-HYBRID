static void termsig_handler(int signum)

{

    state = TERMINATE;

    qemu_notify_event();

}
