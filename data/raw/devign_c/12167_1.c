static int cbs_mpeg2_read_unit(CodedBitstreamContext *ctx,

                               CodedBitstreamUnit *unit)

{

    BitstreamContext bc;

    int err;



    err = bitstream_init(&bc, unit->data, 8 * unit->data_size);

    if (err < 0)

        return err;



    if (MPEG2_START_IS_SLICE(unit->type)) {

        MPEG2RawSlice *slice;

        int pos, len;



        slice = av_mallocz(sizeof(*slice));

        if (!slice)

            return AVERROR(ENOMEM);

        err = cbs_mpeg2_read_slice_header(ctx, &bc, &slice->header);

        if (err < 0) {

            av_free(slice);

            return err;

        }



        pos = bitstream_tell(&bc);

        len = unit->data_size;



        slice->data_size = len - pos / 8;

        slice->data = av_malloc(slice->data_size);

        if (!slice->data) {

            av_free(slice);

            return AVERROR(ENOMEM);

        }



        memcpy(slice->data,

               unit->data + pos / 8, slice->data_size);

        slice->data_bit_start = pos % 8;



        unit->content = slice;



    } else {

        switch (unit->type) {

#define START(start_code, type, func) \

        case start_code: \

            { \

                type *header; \

                header = av_mallocz(sizeof(*header)); \

                if (!header) \

                    return AVERROR(ENOMEM); \

                err = cbs_mpeg2_read_ ## func(ctx, &bc, header); \

                if (err < 0) { \

                    av_free(header); \

                    return err; \

                } \

                unit->content = header; \

            } \

            break;

            START(0x00, MPEG2RawPictureHeader,  picture_header);

            START(0xb2, MPEG2RawUserData,       user_data);

            START(0xb3, MPEG2RawSequenceHeader, sequence_header);

            START(0xb5, MPEG2RawExtensionData,  extension_data);

            START(0xb8, MPEG2RawGroupOfPicturesHeader, group_of_pictures_header);

#undef START

        default:

            av_log(ctx->log_ctx, AV_LOG_ERROR, "Unknown start code %02x.\n",

                   unit->type);

            return AVERROR_INVALIDDATA;

        }

    }



    return 0;

}
