void parse_options(void *optctx, int argc, char **argv, const OptionDef *options,

                   void (*parse_arg_function)(void *, const char*))

{

    const char *opt;

    int optindex, handleoptions = 1, ret;



    /* perform system-dependent conversions for arguments list */

    prepare_app_arguments(&argc, &argv);



    /* parse options */

    optindex = 1;

    while (optindex < argc) {

        opt = argv[optindex++];



        if (handleoptions && opt[0] == '-' && opt[1] != '\0') {

            if (opt[1] == '-' && opt[2] == '\0') {

                handleoptions = 0;

                continue;

            }

            opt++;



            if ((ret = parse_option(optctx, opt, argv[optindex], options)) < 0)

                exit(1);

            optindex += ret;

        } else {

            if (parse_arg_function)

                parse_arg_function(optctx, opt);

        }

    }

}
