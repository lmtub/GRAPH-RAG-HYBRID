static void readline_completion(ReadLineState *rs)

{

    Monitor *mon = cur_mon;

    int len, i, j, max_width, nb_cols, max_prefix;

    char *cmdline;



    rs->nb_completions = 0;



    cmdline = g_malloc(rs->cmd_buf_index + 1);

    memcpy(cmdline, rs->cmd_buf, rs->cmd_buf_index);

    cmdline[rs->cmd_buf_index] = '\0';

    rs->completion_finder(cmdline);

    g_free(cmdline);



    /* no completion found */

    if (rs->nb_completions <= 0)

        return;

    if (rs->nb_completions == 1) {

        len = strlen(rs->completions[0]);

        for(i = rs->completion_index; i < len; i++) {

            readline_insert_char(rs, rs->completions[0][i]);


        /* extra space for next argument. XXX: make it more generic */

        if (len > 0 && rs->completions[0][len - 1] != '/')

            readline_insert_char(rs, ' ');

    } else {

        monitor_printf(mon, "\n");

        max_width = 0;

        max_prefix = 0;	

        for(i = 0; i < rs->nb_completions; i++) {

            len = strlen(rs->completions[i]);

            if (i==0) {

                max_prefix = len;

            } else {

                if (len < max_prefix)

                    max_prefix = len;

                for(j=0; j<max_prefix; j++) {

                    if (rs->completions[i][j] != rs->completions[0][j])

                        max_prefix = j;



            if (len > max_width)

                max_width = len;


        if (max_prefix > 0) 

            for(i = rs->completion_index; i < max_prefix; i++) {

                readline_insert_char(rs, rs->completions[0][i]);


        max_width += 2;

        if (max_width < 10)

            max_width = 10;

        else if (max_width > 80)

            max_width = 80;

        nb_cols = 80 / max_width;

        j = 0;

        for(i = 0; i < rs->nb_completions; i++) {

            monitor_printf(rs->mon, "%-*s", max_width, rs->completions[i]);

            if (++j == nb_cols || i == (rs->nb_completions - 1)) {

                monitor_printf(rs->mon, "\n");

                j = 0;



        readline_show_prompt(rs);




