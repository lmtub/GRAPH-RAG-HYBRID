static inline int parse_nal_units(AVCodecParserContext *s,

                                  AVCodecContext *avctx,

                                  const uint8_t *buf, int buf_size)

{

    H264ParseContext *p = s->priv_data;

    const uint8_t *buf_end = buf + buf_size;



    H2645NAL nal = { NULL };



    unsigned int pps_id;

    unsigned int slice_type;

    int state = -1, got_reset = 0;

    int field_poc[2];

    int ret;



    /* set some sane default values */

    s->pict_type         = AV_PICTURE_TYPE_I;

    s->key_frame         = 0;

    s->picture_structure = AV_PICTURE_STRUCTURE_UNKNOWN;



    ff_h264_sei_uninit(&p->sei);



    if (!buf_size)

        return 0;



    for (;;) {

        const SPS *sps;

        int src_length, consumed;

        buf = avpriv_find_start_code(buf, buf_end, &state);

        if (buf >= buf_end)

            break;

        --buf;

        src_length = buf_end - buf;

        switch (state & 0x1f) {

        case H264_NAL_SLICE:

        case H264_NAL_IDR_SLICE:

            // Do not walk the whole buffer just to decode slice header

            if ((state & 0x1f) == H264_NAL_IDR_SLICE || ((state >> 5) & 0x3) == 0) {

                /* IDR or disposable slice

                 * No need to decode many bytes because MMCOs shall not be present. */

                if (src_length > 60)

                    src_length = 60;

            } else {

                /* To decode up to MMCOs */

                if (src_length > 1000)

                    src_length = 1000;

            }

            break;

        }



        consumed = ff_h2645_extract_rbsp(buf, src_length, &nal);

        if (consumed < 0)

            break;



        ret = init_get_bits(&nal.gb, nal.data, nal.size * 8);

        if (ret < 0)

            goto fail;

        get_bits1(&nal.gb);

        nal.ref_idc = get_bits(&nal.gb, 2);

        nal.type    = get_bits(&nal.gb, 5);



        switch (nal.type) {

        case H264_NAL_SPS:

            ff_h264_decode_seq_parameter_set(&nal.gb, avctx, &p->ps);

            break;

        case H264_NAL_PPS:

            ff_h264_decode_picture_parameter_set(&nal.gb, avctx, &p->ps,

                                                 nal.size_bits);

            break;

        case H264_NAL_SEI:

            ff_h264_sei_decode(&p->sei, &nal.gb, &p->ps, avctx);

            break;

        case H264_NAL_IDR_SLICE:

            s->key_frame = 1;



            p->poc.prev_frame_num        = 0;

            p->poc.prev_frame_num_offset = 0;

            p->poc.prev_poc_msb          =

            p->poc.prev_poc_lsb          = 0;

        /* fall through */

        case H264_NAL_SLICE:

            get_ue_golomb(&nal.gb);  // skip first_mb_in_slice

            slice_type   = get_ue_golomb_31(&nal.gb);

            s->pict_type = ff_h264_golomb_to_pict_type[slice_type % 5];

            if (p->sei.recovery_point.recovery_frame_cnt >= 0) {

                /* key frame, since recovery_frame_cnt is set */

                s->key_frame = 1;

            }

            pps_id = get_ue_golomb(&nal.gb);

            if (pps_id >= MAX_PPS_COUNT) {

                av_log(avctx, AV_LOG_ERROR,

                       "pps_id %u out of range\n", pps_id);

                goto fail;

            }

            if (!p->ps.pps_list[pps_id]) {

                av_log(avctx, AV_LOG_ERROR,

                       "non-existing PPS %u referenced\n", pps_id);

                goto fail;

            }

            p->ps.pps = (const PPS*)p->ps.pps_list[pps_id]->data;

            if (!p->ps.sps_list[p->ps.pps->sps_id]) {

                av_log(avctx, AV_LOG_ERROR,

                       "non-existing SPS %u referenced\n", p->ps.pps->sps_id);

                goto fail;

            }

            p->ps.sps = (SPS*)p->ps.sps_list[p->ps.pps->sps_id]->data;



            sps = p->ps.sps;



            p->poc.frame_num = get_bits(&nal.gb, sps->log2_max_frame_num);



            s->coded_width  = 16 * sps->mb_width;

            s->coded_height = 16 * sps->mb_height;

            s->width        = s->coded_width  - (sps->crop_right + sps->crop_left);

            s->height       = s->coded_height - (sps->crop_top   + sps->crop_bottom);

            if (s->width <= 0 || s->height <= 0) {

                s->width  = s->coded_width;

                s->height = s->coded_height;

            }



            switch (sps->bit_depth_luma) {

            case 9:

                if (sps->chroma_format_idc == 3)      s->format = AV_PIX_FMT_YUV444P9;

                else if (sps->chroma_format_idc == 2) s->format = AV_PIX_FMT_YUV422P9;

                else                                  s->format = AV_PIX_FMT_YUV420P9;

                break;

            case 10:

                if (sps->chroma_format_idc == 3)      s->format = AV_PIX_FMT_YUV444P10;

                else if (sps->chroma_format_idc == 2) s->format = AV_PIX_FMT_YUV422P10;

                else                                  s->format = AV_PIX_FMT_YUV420P10;

                break;

            case 8:

                if (sps->chroma_format_idc == 3)      s->format = AV_PIX_FMT_YUV444P;

                else if (sps->chroma_format_idc == 2) s->format = AV_PIX_FMT_YUV422P;

                else                                  s->format = AV_PIX_FMT_YUV420P;

                break;

            default:

                s->format = AV_PIX_FMT_NONE;

            }



            avctx->profile = ff_h264_get_profile(sps);

            avctx->level   = sps->level_idc;



            if (sps->frame_mbs_only_flag) {

                p->picture_structure = PICT_FRAME;

            } else {

                if (get_bits1(&nal.gb)) { // field_pic_flag

                    p->picture_structure = PICT_TOP_FIELD + get_bits1(&nal.gb); // bottom_field_flag

                } else {

                    p->picture_structure = PICT_FRAME;

                }

            }



            if (nal.type == H264_NAL_IDR_SLICE)

                get_ue_golomb(&nal.gb); /* idr_pic_id */

            if (sps->poc_type == 0) {

                p->poc.poc_lsb = get_bits(&nal.gb, sps->log2_max_poc_lsb);



                if (p->ps.pps->pic_order_present == 1 &&

                    p->picture_structure == PICT_FRAME)

                    p->poc.delta_poc_bottom = get_se_golomb(&nal.gb);

            }



            if (sps->poc_type == 1 &&

                !sps->delta_pic_order_always_zero_flag) {

                p->poc.delta_poc[0] = get_se_golomb(&nal.gb);



                if (p->ps.pps->pic_order_present == 1 &&

                    p->picture_structure == PICT_FRAME)

                    p->poc.delta_poc[1] = get_se_golomb(&nal.gb);

            }



            /* Decode POC of this picture.

             * The prev_ values needed for decoding POC of the next picture are not set here. */

            field_poc[0] = field_poc[1] = INT_MAX;

            ff_h264_init_poc(field_poc, &s->output_picture_number, sps,

                             &p->poc, p->picture_structure, nal.ref_idc);



            /* Continue parsing to check if MMCO_RESET is present.

             * FIXME: MMCO_RESET could appear in non-first slice.

             *        Maybe, we should parse all undisposable non-IDR slice of this

             *        picture until encountering MMCO_RESET in a slice of it. */

            if (nal.ref_idc && nal.type != H264_NAL_IDR_SLICE) {

                got_reset = scan_mmco_reset(s, &nal.gb, avctx);

                if (got_reset < 0)

                    goto fail;

            }



            /* Set up the prev_ values for decoding POC of the next picture. */

            p->poc.prev_frame_num        = got_reset ? 0 : p->poc.frame_num;

            p->poc.prev_frame_num_offset = got_reset ? 0 : p->poc.frame_num_offset;

            if (nal.ref_idc != 0) {

                if (!got_reset) {

                    p->poc.prev_poc_msb = p->poc.poc_msb;

                    p->poc.prev_poc_lsb = p->poc.poc_lsb;

                } else {

                    p->poc.prev_poc_msb = 0;

                    p->poc.prev_poc_lsb =

                        p->picture_structure == PICT_BOTTOM_FIELD ? 0 : field_poc[0];

                }

            }



            if (sps->pic_struct_present_flag) {

                switch (p->sei.picture_timing.pic_struct) {

                case SEI_PIC_STRUCT_TOP_FIELD:

                case SEI_PIC_STRUCT_BOTTOM_FIELD:

                    s->repeat_pict = 0;

                    break;

                case SEI_PIC_STRUCT_FRAME:

                case SEI_PIC_STRUCT_TOP_BOTTOM:

                case SEI_PIC_STRUCT_BOTTOM_TOP:

                    s->repeat_pict = 1;

                    break;

                case SEI_PIC_STRUCT_TOP_BOTTOM_TOP:

                case SEI_PIC_STRUCT_BOTTOM_TOP_BOTTOM:

                    s->repeat_pict = 2;

                    break;

                case SEI_PIC_STRUCT_FRAME_DOUBLING:

                    s->repeat_pict = 3;

                    break;

                case SEI_PIC_STRUCT_FRAME_TRIPLING:

                    s->repeat_pict = 5;

                    break;

                default:

                    s->repeat_pict = p->picture_structure == PICT_FRAME ? 1 : 0;

                    break;

                }

            } else {

                s->repeat_pict = p->picture_structure == PICT_FRAME ? 1 : 0;

            }



            if (p->picture_structure == PICT_FRAME) {

                s->picture_structure = AV_PICTURE_STRUCTURE_FRAME;

                if (sps->pic_struct_present_flag) {

                    switch (p->sei.picture_timing.pic_struct) {

                    case SEI_PIC_STRUCT_TOP_BOTTOM:

                    case SEI_PIC_STRUCT_TOP_BOTTOM_TOP:

                        s->field_order = AV_FIELD_TT;

                        break;

                    case SEI_PIC_STRUCT_BOTTOM_TOP:

                    case SEI_PIC_STRUCT_BOTTOM_TOP_BOTTOM:

                        s->field_order = AV_FIELD_BB;

                        break;

                    default:

                        s->field_order = AV_FIELD_PROGRESSIVE;

                        break;

                    }

                } else {

                    if (field_poc[0] < field_poc[1])

                        s->field_order = AV_FIELD_TT;

                    else if (field_poc[0] > field_poc[1])

                        s->field_order = AV_FIELD_BB;

                    else

                        s->field_order = AV_FIELD_PROGRESSIVE;

                }

            } else {

                if (p->picture_structure == PICT_TOP_FIELD)

                    s->picture_structure = AV_PICTURE_STRUCTURE_TOP_FIELD;

                else

                    s->picture_structure = AV_PICTURE_STRUCTURE_BOTTOM_FIELD;

                s->field_order = AV_FIELD_UNKNOWN;

            }



            av_freep(&nal.rbsp_buffer);

            return 0; /* no need to evaluate the rest */

        }

        buf += consumed;

    }

    /* didn't find a picture! */

    av_log(avctx, AV_LOG_ERROR, "missing picture in access unit\n");

fail:

    av_freep(&nal.rbsp_buffer);

    return -1;

}
