int ff_adts_write_frame_header(ADTSContext *ctx,

                               uint8_t *buf, int size, int pce_size)

{

    PutBitContext pb;



    init_put_bits(&pb, buf, ADTS_HEADER_SIZE);



    /* adts_fixed_header */

    put_bits(&pb, 12, 0xfff);   /* syncword */

    put_bits(&pb, 1, 0);        /* ID */

    put_bits(&pb, 2, 0);        /* layer */

    put_bits(&pb, 1, 1);        /* protection_absent */

    put_bits(&pb, 2, ctx->objecttype); /* profile_objecttype */

    put_bits(&pb, 4, ctx->sample_rate_index);

    put_bits(&pb, 1, 0);        /* private_bit */

    put_bits(&pb, 3, ctx->channel_conf); /* channel_configuration */

    put_bits(&pb, 1, 0);        /* original_copy */

    put_bits(&pb, 1, 0);        /* home */



    /* adts_variable_header */

    put_bits(&pb, 1, 0);        /* copyright_identification_bit */

    put_bits(&pb, 1, 0);        /* copyright_identification_start */

    put_bits(&pb, 13, ADTS_HEADER_SIZE + size + pce_size); /* aac_frame_length */

    put_bits(&pb, 11, 0x7ff);   /* adts_buffer_fullness */

    put_bits(&pb, 2, 0);        /* number_of_raw_data_blocks_in_frame */



    flush_put_bits(&pb);



    return 0;

}
