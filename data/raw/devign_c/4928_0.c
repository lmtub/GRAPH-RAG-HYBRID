static void do_loadvm(int argc, const char **argv)

{

    if (argc != 2) {

        help_cmd(argv[0]);

        return;

    }

    if (qemu_loadvm(argv[1]) < 0) 

        term_printf("I/O error when loading VM from '%s'\n", argv[1]);

}
