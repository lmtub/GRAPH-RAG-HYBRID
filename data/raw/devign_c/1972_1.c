void add_user_command(char *optarg)

{

    ncmdline++;

    cmdline = realloc(cmdline, ncmdline * sizeof(char *));

    if (!cmdline) {

        perror("realloc");

        exit(1);

    }

    cmdline[ncmdline-1] = optarg;

}
