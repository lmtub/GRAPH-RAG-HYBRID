int dpy_set_ui_info(QemuConsole *con, QemuUIInfo *info)

{

    assert(con != NULL);

    con->ui_info = *info;

    if (!con->hw_ops->ui_info) {

        return -1;

    }



    /*

     * Typically we get a flood of these as the user resizes the window.

     * Wait until the dust has settled (one second without updates), then

     * go notify the guest.

     */

    timer_mod(con->ui_timer, qemu_clock_get_ms(QEMU_CLOCK_REALTIME) + 1000);

    return 0;

}
