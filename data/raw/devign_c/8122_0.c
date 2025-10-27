static int cbs_read_ue_golomb(CodedBitstreamContext *ctx, BitstreamContext *bc,

                              const char *name, uint32_t *write_to,

                              uint32_t range_min, uint32_t range_max)

{

    uint32_t value;

    int position;



    if (ctx->trace_enable) {

        char bits[65];

        unsigned int k;

        int i, j;



        position = bitstream_tell(bc);



        for (i = 0; i < 32; i++) {

            k = bitstream_read_bit(bc);

            bits[i] = k ? '1' : '0';

            if (k)

                break;

        }

        if (i >= 32) {

            av_log(ctx->log_ctx, AV_LOG_ERROR, "Invalid ue-golomb "

                   "code found while reading %s: "

                   "more than 31 zeroes.\n", name);

            return AVERROR_INVALIDDATA;

        }

        value = 1;

        for (j = 0; j < i; j++) {

            k = bitstream_read_bit(bc);

            bits[i + j + 1] = k ? '1' : '0';

            value = value << 1 | k;

        }

        bits[i + j + 1] = 0;

        --value;



        ff_cbs_trace_syntax_element(ctx, position, name, bits, value);

    } else {

        value = get_ue_golomb_long(bc);

    }



    if (value < range_min || value > range_max) {

        av_log(ctx->log_ctx, AV_LOG_ERROR, "%s out of range: "

               "%"PRIu32", but must be in [%"PRIu32",%"PRIu32"].\n",

               name, value, range_min, range_max);

        return AVERROR_INVALIDDATA;

    }



    *write_to = value;

    return 0;

}
