static int parse_args(int argc, char **argv)

{

    const char *r;

    int optind;

    struct qemu_argument *arginfo;



    for (arginfo = arg_table; arginfo->handle_opt != NULL; arginfo++) {

        if (arginfo->env == NULL) {

            continue;

        }



        r = getenv(arginfo->env);

        if (r != NULL) {

            arginfo->handle_opt(r);

        }

    }



    optind = 1;

    for (;;) {

        if (optind >= argc) {

            break;

        }

        r = argv[optind];

        if (r[0] != '-') {

            break;

        }

        optind++;

        r++;

        if (!strcmp(r, "-")) {

            break;

        }



        for (arginfo = arg_table; arginfo->handle_opt != NULL; arginfo++) {

            if (!strcmp(r, arginfo->argv)) {

                if (optind >= argc) {

                    usage();

                }



                arginfo->handle_opt(argv[optind]);



                if (arginfo->has_arg) {

                    optind++;

                }



                break;

            }

        }



        /* no option matched the current argv */

        if (arginfo->handle_opt == NULL) {

            usage();

        }

    }



    if (optind >= argc) {

        usage();

    }



    filename = argv[optind];

    exec_path = argv[optind];



    return optind;

}
