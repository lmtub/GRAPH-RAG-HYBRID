static int setup_common(char *argv[], int argv_sz)

{

    memset(cur_ide, 0, sizeof(cur_ide));

    return append_arg(0, argv, argv_sz,

                      g_strdup("-nodefaults -display none"));

}
