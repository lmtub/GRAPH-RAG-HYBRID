static int img_create(int argc, char **argv)

{

    int c, ret = 0;

    uint64_t img_size = -1;

    const char *fmt = "raw";

    const char *base_fmt = NULL;

    const char *filename;

    const char *base_filename = NULL;

    char *options = NULL;



    for(;;) {

        c = getopt(argc, argv, "F:b:f:he6o:");

        if (c == -1) {

            break;

        }

        switch(c) {

        case '?':

        case 'h':

            help();

            break;

        case 'F':

            base_fmt = optarg;

            break;

        case 'b':

            base_filename = optarg;

            break;

        case 'f':

            fmt = optarg;

            break;

        case 'e':

            error_report("option -e is deprecated, please use \'-o "

                  "encryption\' instead!");

            return 1;

        case '6':

            error_report("option -6 is deprecated, please use \'-o "

                  "compat6\' instead!");

            return 1;

        case 'o':

            options = optarg;

            break;

        }

    }



    /* Get the filename */

    if (optind >= argc) {

        help();

    }

    filename = argv[optind++];



    /* Get image size, if specified */

    if (optind < argc) {

        int64_t sval;

        char *end;

        sval = strtosz_suffix(argv[optind++], &end, STRTOSZ_DEFSUFFIX_B);

        if (sval < 0 || *end) {

            error_report("Invalid image size specified! You may use k, M, G or "

                  "T suffixes for ");

            error_report("kilobytes, megabytes, gigabytes and terabytes.");

            ret = -1;

            goto out;

        }

        img_size = (uint64_t)sval;

    }



    if (options && !strcmp(options, "?")) {

        ret = print_block_option_help(filename, fmt);

        goto out;

    }



    ret = bdrv_img_create(filename, fmt, base_filename, base_fmt,

                          options, img_size, BDRV_O_FLAGS);

out:

    if (ret) {

        return 1;

    }

    return 0;

}
