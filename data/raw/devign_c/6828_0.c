static void term_hist_add(const char *cmdline)

{

    char *hist_entry, *new_entry;

    int idx;



    if (cmdline[0] == '\0')

	return;

    new_entry = NULL;

    if (term_hist_entry != -1) {

	/* We were editing an existing history entry: replace it */

	hist_entry = term_history[term_hist_entry];

	idx = term_hist_entry;

	if (strcmp(hist_entry, cmdline) == 0) {

	    goto same_entry;

	}

    }

    /* Search cmdline in history buffers */

    for (idx = 0; idx < TERM_MAX_CMDS; idx++) {

	hist_entry = term_history[idx];

	if (hist_entry == NULL)

	    break;

	if (strcmp(hist_entry, cmdline) == 0) {

	same_entry:

	    new_entry = hist_entry;

	    /* Put this entry at the end of history */

	    memmove(&term_history[idx], &term_history[idx + 1],

		    &term_history[TERM_MAX_CMDS] - &term_history[idx + 1]);

	    term_history[TERM_MAX_CMDS - 1] = NULL;

	    for (; idx < TERM_MAX_CMDS; idx++) {

		if (term_history[idx] == NULL)

		    break;

	    }

	    break;

	}

    }

    if (idx == TERM_MAX_CMDS) {

	/* Need to get one free slot */

	free(term_history[0]);

	memcpy(term_history, &term_history[1],

	       &term_history[TERM_MAX_CMDS] - &term_history[1]);

	term_history[TERM_MAX_CMDS - 1] = NULL;

	idx = TERM_MAX_CMDS - 1;

    }

    if (new_entry == NULL)

	new_entry = strdup(cmdline);

    term_history[idx] = new_entry;

    term_hist_entry = -1;

}
