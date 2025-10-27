static void cmd_args_init(CmdArgs *cmd_args)

{

    cmd_args->name = qstring_new();

    cmd_args->type = cmd_args->flag = cmd_args->optional = 0;

}
