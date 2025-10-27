static void monitor_find_completion(const char *cmdline)

{

    const char *cmdname;

    char *args[MAX_ARGS];

    int nb_args, i, len;

    const char *ptype, *str;

    const mon_cmd_t *cmd;

    const KeyDef *key;



    parse_cmdline(cmdline, &nb_args, args);

#ifdef DEBUG_COMPLETION

    for(i = 0; i < nb_args; i++) {

        monitor_printf(cur_mon, "arg%d = '%s'\n", i, (char *)args[i]);

    }

#endif



    /* if the line ends with a space, it means we want to complete the

       next arg */

    len = strlen(cmdline);

    if (len > 0 && qemu_isspace(cmdline[len - 1])) {

        if (nb_args >= MAX_ARGS)

            return;

        args[nb_args++] = qemu_strdup("");

    }

    if (nb_args <= 1) {

        /* command completion */

        if (nb_args == 0)

            cmdname = "";

        else

            cmdname = args[0];

        readline_set_completion_index(cur_mon->rs, strlen(cmdname));

        for(cmd = mon_cmds; cmd->name != NULL; cmd++) {

            cmd_completion(cmdname, cmd->name);

        }

    } else {

        /* find the command */

        for(cmd = mon_cmds; cmd->name != NULL; cmd++) {

            if (compare_cmd(args[0], cmd->name))

                goto found;

        }

        return;

    found:

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

            ptype += 2;

        }

        switch(*ptype) {

        case 'F':

            /* file completion */

            readline_set_completion_index(cur_mon->rs, strlen(str));

            file_completion(str);

            break;

        case 'B':

            /* block device name completion */

            readline_set_completion_index(cur_mon->rs, strlen(str));

            bdrv_iterate(block_completion_it, (void *)str);

            break;

        case 's':

            /* XXX: more generic ? */

            if (!strcmp(cmd->name, "info")) {

                readline_set_completion_index(cur_mon->rs, strlen(str));

                for(cmd = info_cmds; cmd->name != NULL; cmd++) {

                    cmd_completion(str, cmd->name);

                }

            } else if (!strcmp(cmd->name, "sendkey")) {

                char *sep = strrchr(str, '-');

                if (sep)

                    str = sep + 1;

                readline_set_completion_index(cur_mon->rs, strlen(str));

                for(key = key_defs; key->name != NULL; key++) {

                    cmd_completion(str, key->name);

                }

            } else if (!strcmp(cmd->name, "help|?")) {

                readline_set_completion_index(cur_mon->rs, strlen(str));

                for (cmd = mon_cmds; cmd->name != NULL; cmd++) {

                    cmd_completion(str, cmd->name);

                }

            }

            break;

        default:

            break;

        }

    }

    for(i = 0; i < nb_args; i++)

        qemu_free(args[i]);

}
