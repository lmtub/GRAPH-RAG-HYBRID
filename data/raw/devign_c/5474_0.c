void do_info_vnc(Monitor *mon)

{

    if (vnc_display == NULL || vnc_display->display == NULL)

        monitor_printf(mon, "VNC server disabled\n");

    else {

        monitor_printf(mon, "VNC server active on: ");

        monitor_print_filename(mon, vnc_display->display);

        monitor_printf(mon, "\n");



	if (vnc_display->clients == NULL)

            monitor_printf(mon, "No client connected\n");

	else

	    monitor_printf(mon, "Client connected\n");

    }

}
