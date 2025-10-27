static void vc1_extract_headers(AVCodecParserContext *s, AVCodecContext *avctx,

                                const uint8_t *buf, int buf_size)

{

    VC1ParseContext *vpc = s->priv_data;

    GetBitContext gb;

    const uint8_t *start, *end, *next;

    uint8_t *buf2 = av_mallocz(buf_size + FF_INPUT_BUFFER_PADDING_SIZE);



    vpc->v.s.avctx = avctx;

    vpc->v.parse_only = 1;

    vpc->v.first_pic_header_flag = 1;

    next = buf;

    s->repeat_pict = 0;



    for(start = buf, end = buf + buf_size; next < end; start = next){

        int buf2_size, size;



        next = find_next_marker(start + 4, end);

        size = next - start - 4;

        buf2_size = vc1_unescape_buffer(start + 4, size, buf2);

        init_get_bits(&gb, buf2, buf2_size * 8);

        if(size <= 0) continue;

        switch(AV_RB32(start)){

        case VC1_CODE_SEQHDR:

            ff_vc1_decode_sequence_header(avctx, &vpc->v, &gb);

            break;

        case VC1_CODE_ENTRYPOINT:

            ff_vc1_decode_entry_point(avctx, &vpc->v, &gb);

            break;

        case VC1_CODE_FRAME:

            if(vpc->v.profile < PROFILE_ADVANCED)

                ff_vc1_parse_frame_header    (&vpc->v, &gb);

            else

                ff_vc1_parse_frame_header_adv(&vpc->v, &gb);



            /* keep AV_PICTURE_TYPE_BI internal to VC1 */

            if (vpc->v.s.pict_type == AV_PICTURE_TYPE_BI)

                s->pict_type = AV_PICTURE_TYPE_B;

            else

                s->pict_type = vpc->v.s.pict_type;



            if (avctx->ticks_per_frame > 1){

                // process pulldown flags

                s->repeat_pict = 1;

                // Pulldown flags are only valid when 'broadcast' has been set.

                // So ticks_per_frame will be 2

                if (vpc->v.rff){

                    // repeat field

                    s->repeat_pict = 2;

                }else if (vpc->v.rptfrm){

                    // repeat frames

                    s->repeat_pict = vpc->v.rptfrm * 2 + 1;

                }

            }



            if (vpc->v.broadcast && vpc->v.interlace && !vpc->v.psf)

                s->field_order = vpc->v.tff ? AV_FIELD_TT : AV_FIELD_BB;

            else

                s->field_order = AV_FIELD_PROGRESSIVE;



            break;

        }

    }



    av_free(buf2);

}
