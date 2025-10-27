static void monitor_find_completion(Monitor *mon,

                                    const char *cmdline)

{

    const char *cmdname;

    char *args[MAX_ARGS];

    int nb_args, i, len;

    const char *ptype, *str;

    const mon_cmd_t *cmd;

    MonitorBlockComplete mbs;



    parse_cmdline(cmdline, &nb_args, args);

#ifdef DEBUG_COMPLETION

    for (i = 0; i < nb_args; i++) {

        monitor_printf(mon, "arg%d = '%s'\n", i, args[i]);

    }

#endif



    /* if the line ends with a space, it means we want to complete the

       next arg */

    len = strlen(cmdline);

    if (len > 0 && qemu_isspace(cmdline[len - 1])) {

        if (nb_args >= MAX_ARGS) {

            goto cleanup;

        }

        args[nb_args++] = g_strdup("");

    }

    if (nb_args <= 1) {

        /* command completion */

        if (nb_args == 0)

            cmdname = "";

        else

            cmdname = args[0];

        readline_set_completion_index(mon->rs, strlen(cmdname));

        for (cmd = mon->cmd_table; cmd->name != NULL; cmd++) {

            cmd_completion(mon, cmdname, cmd->name);

        }

    } else {

        /* find the command */

        for (cmd = mon->cmd_table; cmd->name != NULL; cmd++) {

            if (compare_cmd(args[0], cmd->name)) {

                break;

            }

        }

        if (!cmd->name) {

            goto cleanup;

        }



        ptype = next_arg_type(cmd->args_type);

        for(i = 0; i < nb_args - 2; i++) {

            if (*ptype != '\0') {

                ptype = next_arg_type(ptype);

                while (*ptype == '?')

                    ptype = next_arg_type(ptype);

            }

        }

        str = args[nb_args - 1];

        if (*ptype == '-' && ptype[1] != '\0') {

            ptype = next_arg_type(ptype);

        }

        switch(*ptype) {

        case 'F':

            /* file completion */

            readline_set_completion_index(mon->rs, strlen(str));

            file_completion(mon, str);

            break;

        case 'B':

            /* block device name completion */

            mbs.mon = mon;

            mbs.input = str;

            readline_set_completion_index(mon->rs, strlen(str));

            bdrv_iterate(block_completion_it, &mbs);

            break;

        case 's':

            /* XXX: more generic ? */

            if (!strcmp(cmd->name, "info")) {

                readline_set_completion_index(mon->rs, strlen(str));

                for(cmd = info_cmds; cmd->name != NULL; cmd++) {

                    cmd_completion(mon, str, cmd->name);

                }

            } else if (!strcmp(cmd->name, "sendkey")) {

                char *sep = strrchr(str, '-');

                if (sep)

                    str = sep + 1;

                readline_set_completion_index(mon->rs, strlen(str));

                for (i = 0; i < Q_KEY_CODE_MAX; i++) {

                    cmd_completion(mon, str, QKeyCode_lookup[i]);

                }

            } else if (!strcmp(cmd->name, "help|?")) {

                readline_set_completion_index(mon->rs, strlen(str));

                for (cmd = mon->cmd_table; cmd->name != NULL; cmd++) {

                    cmd_completion(mon, str, cmd->name);

                }

            }

            break;

        default:

            break;

        }

    }



cleanup:

    for (i = 0; i < nb_args; i++) {

        g_free(args[i]);

    }

}
