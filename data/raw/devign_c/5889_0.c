static void monitor_start_input(void)

{

    readline_start("(qemu) ", 0, monitor_handle_command1, NULL);

}
