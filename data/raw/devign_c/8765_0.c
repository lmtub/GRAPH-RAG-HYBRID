static av_cold int vc1_decode_init(AVCodecContext *avctx)

{

    VC1Context *v = avctx->priv_data;

    MpegEncContext *s = &v->s;

    GetBitContext gb;



    /* save the container output size for WMImage */

    v->output_width  = avctx->width;

    v->output_height = avctx->height;



    if (!avctx->extradata_size || !avctx->extradata)

        return -1;

    if (!(avctx->flags & AV_CODEC_FLAG_GRAY))

        avctx->pix_fmt = ff_get_format(avctx, avctx->codec->pix_fmts);

    else

        avctx->pix_fmt = AV_PIX_FMT_GRAY8;

    v->s.avctx = avctx;



    if (ff_vc1_init_common(v) < 0)

        return -1;

    ff_blockdsp_init(&s->bdsp, avctx);

    ff_h264chroma_init(&v->h264chroma, 8);

    ff_qpeldsp_init(&s->qdsp);



    if (avctx->codec_id == AV_CODEC_ID_WMV3 || avctx->codec_id == AV_CODEC_ID_WMV3IMAGE) {

        int count = 0;



        // looks like WMV3 has a sequence header stored in the extradata

        // advanced sequence header may be before the first frame

        // the last byte of the extradata is a version number, 1 for the

        // samples we can decode



        init_get_bits(&gb, avctx->extradata, avctx->extradata_size*8);



        if (ff_vc1_decode_sequence_header(avctx, v, &gb) < 0)

          return -1;



        count = avctx->extradata_size*8 - get_bits_count(&gb);

        if (count > 0) {

            av_log(avctx, AV_LOG_INFO, "Extra data: %i bits left, value: %X\n",

                   count, get_bits_long(&gb, FFMIN(count, 32)));

        } else if (count < 0) {

            av_log(avctx, AV_LOG_INFO, "Read %i bits in overflow\n", -count);

        }

    } else { // VC1/WVC1/WVP2

        const uint8_t *start = avctx->extradata;

        uint8_t *end = avctx->extradata + avctx->extradata_size;

        const uint8_t *next;

        int size, buf2_size;

        uint8_t *buf2 = NULL;

        int seq_initialized = 0, ep_initialized = 0;



        if (avctx->extradata_size < 16) {

            av_log(avctx, AV_LOG_ERROR, "Extradata size too small: %i\n", avctx->extradata_size);

            return -1;

        }



        buf2  = av_mallocz(avctx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);

        start = find_next_marker(start, end); // in WVC1 extradata first byte is its size, but can be 0 in mkv

        next  = start;

        for (; next < end; start = next) {

            next = find_next_marker(start + 4, end);

            size = next - start - 4;

            if (size <= 0)

                continue;

            buf2_size = vc1_unescape_buffer(start + 4, size, buf2);

            init_get_bits(&gb, buf2, buf2_size * 8);

            switch (AV_RB32(start)) {

            case VC1_CODE_SEQHDR:

                if (ff_vc1_decode_sequence_header(avctx, v, &gb) < 0) {

                    av_free(buf2);

                    return -1;

                }

                seq_initialized = 1;

                break;

            case VC1_CODE_ENTRYPOINT:

                if (ff_vc1_decode_entry_point(avctx, v, &gb) < 0) {

                    av_free(buf2);

                    return -1;

                }

                ep_initialized = 1;

                break;

            }

        }

        av_free(buf2);

        if (!seq_initialized || !ep_initialized) {

            av_log(avctx, AV_LOG_ERROR, "Incomplete extradata\n");

            return -1;

        }

        v->res_sprite = (avctx->codec_id == AV_CODEC_ID_VC1IMAGE);

    }



    v->sprite_output_frame = av_frame_alloc();

    if (!v->sprite_output_frame)

        return AVERROR(ENOMEM);



    avctx->profile = v->profile;

    if (v->profile == PROFILE_ADVANCED)

        avctx->level = v->level;



    avctx->has_b_frames = !!avctx->max_b_frames;



    if (v->color_prim == 1 || v->color_prim == 5 || v->color_prim == 6)

        avctx->color_primaries = v->color_prim;

    if (v->transfer_char == 1 || v->transfer_char == 7)

        avctx->color_trc = v->transfer_char;

    if (v->matrix_coef == 1 || v->matrix_coef == 6 || v->matrix_coef == 7)

        avctx->colorspace = v->matrix_coef;



    s->mb_width  = (avctx->coded_width  + 15) >> 4;

    s->mb_height = (avctx->coded_height + 15) >> 4;



    if (v->profile == PROFILE_ADVANCED || v->res_fasttx) {

        ff_vc1_init_transposed_scantables(v);

    } else {

        memcpy(v->zz_8x8, ff_wmv1_scantable, 4*64);

        v->left_blk_sh = 3;

        v->top_blk_sh  = 0;

    }



    if (avctx->codec_id == AV_CODEC_ID_WMV3IMAGE || avctx->codec_id == AV_CODEC_ID_VC1IMAGE) {

        v->sprite_width  = avctx->coded_width;

        v->sprite_height = avctx->coded_height;



        avctx->coded_width  = avctx->width  = v->output_width;

        avctx->coded_height = avctx->height = v->output_height;



        // prevent 16.16 overflows

        if (v->sprite_width  > 1 << 14 ||

            v->sprite_height > 1 << 14 ||

            v->output_width  > 1 << 14 ||

            v->output_height > 1 << 14) return -1;

    }

    return 0;

}
