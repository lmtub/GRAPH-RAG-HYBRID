static int xvid_ff_2pass_create(xvid_plg_create_t * param,

                                void ** handle) {

    struct xvid_ff_pass1 *x = (struct xvid_ff_pass1 *)param->param;

    char *log = x->context->twopassbuffer;



    /* Do a quick bounds check */

    if( log == NULL )

        return XVID_ERR_FAIL;



    /* We use snprintf() */

    /* This is because we can safely prevent a buffer overflow */

    log[0] = 0;

    snprintf(log, BUFFER_REMAINING(log),

        "# avconv 2-pass log file, using xvid codec\n");

    snprintf(BUFFER_CAT(log), BUFFER_REMAINING(log),

        "# Do not modify. libxvidcore version: %d.%d.%d\n\n",

        XVID_VERSION_MAJOR(XVID_VERSION),

        XVID_VERSION_MINOR(XVID_VERSION),

        XVID_VERSION_PATCH(XVID_VERSION));



    *handle = x->context;

    return 0;

}
