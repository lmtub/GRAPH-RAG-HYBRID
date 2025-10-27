static int check_opt(const CmdArgs *cmd_args, const char *name, QDict *args)

{

    if (!cmd_args->optional) {

        qerror_report(QERR_MISSING_PARAMETER, name);

        return -1;

    }



    if (cmd_args->type == '-') {

        /* handlers expect a value, they need to be changed */

        qdict_put(args, name, qint_from_int(0));

    }



    return 0;

}
