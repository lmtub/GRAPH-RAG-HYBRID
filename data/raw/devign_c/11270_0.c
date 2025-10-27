static void term_completion(void)

{

    int len, i, j, max_width, nb_cols;

    char *cmdline;



    nb_completions = 0;

    

    cmdline = qemu_malloc(term_cmd_buf_index + 1);

    if (!cmdline)

        return;

    memcpy(cmdline, term_cmd_buf, term_cmd_buf_index);

    cmdline[term_cmd_buf_index] = '\0';

    find_completion(cmdline);

    qemu_free(cmdline);



    /* no completion found */

    if (nb_completions <= 0)

        return;

    if (nb_completions == 1) {

        len = strlen(completions[0]);

        for(i = completion_index; i < len; i++) {

            term_insert_char(completions[0][i]);

        }

        /* extra space for next argument. XXX: make it more generic */

        if (len > 0 && completions[0][len - 1] != '/')

            term_insert_char(' ');

    } else {

        term_printf("\n");

        max_width = 0;

        for(i = 0; i < nb_completions; i++) {

            len = strlen(completions[i]);

            if (len > max_width)

                max_width = len;

        }

        max_width += 2;

        if (max_width < 10)

            max_width = 10;

        else if (max_width > 80)

            max_width = 80;

        nb_cols = 80 / max_width;

        j = 0;

        for(i = 0; i < nb_completions; i++) {

            term_printf("%-*s", max_width, completions[i]);

            if (++j == nb_cols || i == (nb_completions - 1)) {

                term_printf("\n");

                j = 0;

            }

        }

        term_show_prompt2();

    }

}
