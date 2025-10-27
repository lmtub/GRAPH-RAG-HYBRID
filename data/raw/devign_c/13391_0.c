static void vaapi_encode_h264_write_sei(PutBitContext *pbc,

                                        VAAPIEncodeContext *ctx,

                                        VAAPIEncodePicture *pic)

{

    VAAPIEncodeH264Context *priv = ctx->priv_data;

    PutBitContext payload_bits;

    char payload[256];

    int payload_type, payload_size, i;

    void (*write_payload)(PutBitContext *pbc,

                          VAAPIEncodeContext *ctx,

                          VAAPIEncodePicture *pic) = NULL;



    vaapi_encode_h264_write_nal_header(pbc, NAL_SEI, 0);



    for (payload_type = 0; payload_type < 64; payload_type++) {

        switch (payload_type) {

        case SEI_TYPE_BUFFERING_PERIOD:

            if (!priv->send_timing_sei ||

                pic->type != PICTURE_TYPE_IDR)

                continue;

            write_payload = &vaapi_encode_h264_write_buffering_period;

            break;

        case SEI_TYPE_PIC_TIMING:

            if (!priv->send_timing_sei)

                continue;

            write_payload = &vaapi_encode_h264_write_pic_timing;

            break;

        case SEI_TYPE_USER_DATA_UNREGISTERED:

            if (pic->encode_order != 0)

                continue;

            write_payload = &vaapi_encode_h264_write_identifier;

            break;

        default:

            continue;

        }



        init_put_bits(&payload_bits, payload, sizeof(payload));

        write_payload(&payload_bits, ctx, pic);

        if (put_bits_count(&payload_bits) & 7) {

            write_u(&payload_bits, 1, 1, bit_equal_to_one);

            while (put_bits_count(&payload_bits) & 7)

                write_u(&payload_bits, 1, 0, bit_equal_to_zero);

        }

        payload_size = put_bits_count(&payload_bits) / 8;

        flush_put_bits(&payload_bits);



        u(8, payload_type, last_payload_type_byte);

        u(8, payload_size, last_payload_size_byte);

        for (i = 0; i < payload_size; i++)

            u(8, payload[i] & 0xff, sei_payload);

    }



    vaapi_encode_h264_write_trailing_rbsp(pbc);

}
