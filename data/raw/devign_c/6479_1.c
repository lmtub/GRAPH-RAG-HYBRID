void do_info_vnc(void)

{

    if (vnc_state == NULL)

	term_printf("VNC server disabled\n");

    else {

	term_printf("VNC server active on: ");

	term_print_filename(vnc_state->display);

	term_printf("\n");



	if (vnc_state->csock == -1)

	    term_printf("No client connected\n");

	else

	    term_printf("Client connected\n");

    }

}
