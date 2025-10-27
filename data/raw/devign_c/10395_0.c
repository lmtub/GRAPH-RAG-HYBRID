static int xvid_ff_2pass_after(struct xvid_context *ref,

                                xvid_plg_data_t *param) {

    char *log = ref->twopassbuffer;

    const char *frame_types = " ipbs";

    char frame_type;



    /* Quick bounds check */

    if( log == NULL )

        return XVID_ERR_FAIL;



    /* Convert the type given to us into a character */

    if( param->type < 5 && param->type > 0 ) {

        frame_type = frame_types[param->type];

    } else {

        return XVID_ERR_FAIL;

    }



    snprintf(BUFFER_CAT(log), BUFFER_REMAINING(log),

        "%c %d %d %d %d %d %d\n",

        frame_type, param->stats.quant, param->stats.kblks, param->stats.mblks,

        param->stats.ublks, param->stats.length, param->stats.hlength);



    return 0;

}
