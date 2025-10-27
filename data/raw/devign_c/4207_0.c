static int monitor_check_qmp_args(const mon_cmd_t *cmd, QDict *args)

{

    int err;

    const char *p;

    CmdArgs cmd_args;

    QemuOptsList *opts_list;



    if (cmd->args_type == NULL) {

        return (qdict_size(args) == 0 ? 0 : -1);

    }



    err = 0;

    cmd_args_init(&cmd_args);

    opts_list = NULL;



    for (p = cmd->args_type;; p++) {

        if (*p == ':') {

            cmd_args.type = *++p;

            p++;

            if (cmd_args.type == '-') {

                cmd_args.flag = *p++;

                cmd_args.optional = 1;

            } else if (cmd_args.type == 'O') {

                opts_list = qemu_find_opts(qstring_get_str(cmd_args.name));

                assert(opts_list);

            } else if (*p == '?') {

                cmd_args.optional = 1;

                p++;

            }



            assert(*p == ',' || *p == '\0');

            if (opts_list) {

                err = check_opts(opts_list, args);

                opts_list = NULL;

            } else {

                err = check_arg(&cmd_args, args);

                QDECREF(cmd_args.name);

                cmd_args_init(&cmd_args);

            }



            if (err < 0) {

                break;

            }

        } else {

            qstring_append_chr(cmd_args.name, *p);

        }



        if (*p == '\0') {

            break;

        }

    }



    QDECREF(cmd_args.name);

    return err;

}
