static int svq3_decode_slice_header(AVCodecContext *avctx)

{

    SVQ3Context *svq3 = avctx->priv_data;

    H264Context *h = &svq3->h;

    MpegEncContext *s = &h->s;

    const int mb_xy = h->mb_xy;

    int i, header;



    header = get_bits(&s->gb, 8);



    if (((header & 0x9F) != 1 && (header & 0x9F) != 2) || (header & 0x60) == 0) {

        /* TODO: what? */

        av_log(avctx, AV_LOG_ERROR, "unsupported slice header (%02X)\n", header);

        return -1;

    } else {

        int length = (header >> 5) & 3;



        svq3->next_slice_index = get_bits_count(&s->gb) + 8*show_bits(&s->gb, 8*length) + 8*length;



        if (svq3->next_slice_index > s->gb.size_in_bits) {

            av_log(avctx, AV_LOG_ERROR, "slice after bitstream end\n");

            return -1;

    }



        s->gb.size_in_bits = svq3->next_slice_index - 8*(length - 1);

        skip_bits(&s->gb, 8);



        if (svq3->watermark_key) {

            uint32_t header = AV_RL32(&s->gb.buffer[(get_bits_count(&s->gb)>>3)+1]);

            AV_WL32(&s->gb.buffer[(get_bits_count(&s->gb)>>3)+1], header ^ svq3->watermark_key);

        }

        if (length > 0) {

            memcpy((uint8_t *) &s->gb.buffer[get_bits_count(&s->gb) >> 3],

                   &s->gb.buffer[s->gb.size_in_bits >> 3], (length - 1));

        }

        skip_bits_long(&s->gb, 0);

    }



    if ((i = svq3_get_ue_golomb(&s->gb)) == INVALID_VLC || i >= 3){

        av_log(h->s.avctx, AV_LOG_ERROR, "illegal slice type %d \n", i);

        return -1;

    }



    h->slice_type = golomb_to_pict_type[i];



    if ((header & 0x9F) == 2) {

        i = (s->mb_num < 64) ? 6 : (1 + av_log2 (s->mb_num - 1));

        s->mb_skip_run = get_bits(&s->gb, i) - (s->mb_x + (s->mb_y * s->mb_width));

    } else {

        skip_bits1(&s->gb);

        s->mb_skip_run = 0;

    }



    h->slice_num = get_bits(&s->gb, 8);

    s->qscale = get_bits(&s->gb, 5);

    s->adaptive_quant = get_bits1(&s->gb);



    /* unknown fields */

    skip_bits1(&s->gb);



    if (svq3->unknown_flag) {

        skip_bits1(&s->gb);

    }



    skip_bits1(&s->gb);

    skip_bits(&s->gb, 2);



    while (get_bits1(&s->gb)) {

        skip_bits(&s->gb, 8);

    }



    /* reset intra predictors and invalidate motion vector references */

    if (s->mb_x > 0) {

        memset(h->intra4x4_pred_mode+h->mb2br_xy[mb_xy - 1      ]+3, -1, 4*sizeof(int8_t));

        memset(h->intra4x4_pred_mode+h->mb2br_xy[mb_xy - s->mb_x]  , -1, 8*sizeof(int8_t)*s->mb_x);

    }

    if (s->mb_y > 0) {

        memset(h->intra4x4_pred_mode+h->mb2br_xy[mb_xy - s->mb_stride], -1, 8*sizeof(int8_t)*(s->mb_width - s->mb_x));



        if (s->mb_x > 0) {

            h->intra4x4_pred_mode[h->mb2br_xy[mb_xy - s->mb_stride - 1]+3] = -1;

        }

    }



    return 0;

}
