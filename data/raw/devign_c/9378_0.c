static void do_info_history (void)

{

    int i;



    for (i = 0; i < TERM_MAX_CMDS; i++) {

	if (term_history[i] == NULL)

	    break;

	term_printf("%d: '%s'\n", i, term_history[i]);

    }

}
