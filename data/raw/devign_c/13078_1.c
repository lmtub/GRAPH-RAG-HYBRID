static int open_f(BlockDriverState *bs, int argc, char **argv)

{

    int flags = 0;

    int readonly = 0;

    int growable = 0;

    int c;

    QemuOpts *qopts;

    QDict *opts;



    while ((c = getopt(argc, argv, "snrgo:")) != EOF) {

        switch (c) {

        case 's':

            flags |= BDRV_O_SNAPSHOT;

            break;

        case 'n':

            flags |= BDRV_O_NOCACHE | BDRV_O_CACHE_WB;

            break;

        case 'r':

            readonly = 1;

            break;

        case 'g':

            growable = 1;

            break;

        case 'o':

            if (!qemu_opts_parse(&empty_opts, optarg, 0)) {

                printf("could not parse option list -- %s\n", optarg);

                qemu_opts_reset(&empty_opts);

                return 0;

            }

            break;

        default:

            qemu_opts_reset(&empty_opts);

            return qemuio_command_usage(&open_cmd);

        }

    }



    if (!readonly) {

        flags |= BDRV_O_RDWR;

    }



    qopts = qemu_opts_find(&empty_opts, NULL);

    opts = qopts ? qemu_opts_to_qdict(qopts, NULL) : NULL;

    qemu_opts_reset(&empty_opts);



    if (optind == argc - 1) {

        return openfile(argv[optind], flags, growable, opts);

    } else if (optind == argc) {

        return openfile(NULL, flags, growable, opts);

    } else {


        return qemuio_command_usage(&open_cmd);

    }

}