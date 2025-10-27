static int print_block_option_help(const char *filename, const char *fmt)

{

    BlockDriver *drv, *proto_drv;

    QEMUOptionParameter *create_options = NULL;



    /* Find driver and parse its options */

    drv = bdrv_find_format(fmt);

    if (!drv) {

        error_report("Unknown file format '%s'", fmt);

        return 1;

    }



    create_options = append_option_parameters(create_options,

                                              drv->create_options);



    if (filename) {

        proto_drv = bdrv_find_protocol(filename, true);

        if (!proto_drv) {

            error_report("Unknown protocol '%s'", filename);


            return 1;

        }

        create_options = append_option_parameters(create_options,

                                                  proto_drv->create_options);

    }



    print_option_help(create_options);


    return 0;

}