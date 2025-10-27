static int estimate_best_b_count(MpegEncContext *s)

{

    AVCodec *codec    = avcodec_find_encoder(s->avctx->codec_id);

    AVCodecContext *c = avcodec_alloc_context3(NULL);

    AVFrame input[FF_MAX_B_FRAMES + 2];

    const int scale = s->avctx->brd_scale;

    int i, j, out_size, p_lambda, b_lambda, lambda2;

    int64_t best_rd  = INT64_MAX;

    int best_b_count = -1;



    assert(scale >= 0 && scale <= 3);



    //emms_c();

    //s->next_picture_ptr->quality;

    p_lambda = s->last_lambda_for[AV_PICTURE_TYPE_P];

    //p_lambda * FFABS(s->avctx->b_quant_factor) + s->avctx->b_quant_offset;

    b_lambda = s->last_lambda_for[AV_PICTURE_TYPE_B];

    if (!b_lambda) // FIXME we should do this somewhere else

        b_lambda = p_lambda;

    lambda2  = (b_lambda * b_lambda + (1 << FF_LAMBDA_SHIFT) / 2) >>

               FF_LAMBDA_SHIFT;



    c->width        = s->width  >> scale;

    c->height       = s->height >> scale;

    c->flags        = CODEC_FLAG_QSCALE | CODEC_FLAG_PSNR |

                      CODEC_FLAG_INPUT_PRESERVED /*| CODEC_FLAG_EMU_EDGE*/;

    c->flags       |= s->avctx->flags & CODEC_FLAG_QPEL;

    c->mb_decision  = s->avctx->mb_decision;

    c->me_cmp       = s->avctx->me_cmp;

    c->mb_cmp       = s->avctx->mb_cmp;

    c->me_sub_cmp   = s->avctx->me_sub_cmp;

    c->pix_fmt      = AV_PIX_FMT_YUV420P;

    c->time_base    = s->avctx->time_base;

    c->max_b_frames = s->max_b_frames;



    if (avcodec_open2(c, codec, NULL) < 0)

        return -1;



    for (i = 0; i < s->max_b_frames + 2; i++) {

        int ysize = c->width * c->height;

        int csize = (c->width / 2) * (c->height / 2);

        Picture pre_input, *pre_input_ptr = i ? s->input_picture[i - 1] :

                                                s->next_picture_ptr;



        avcodec_get_frame_defaults(&input[i]);

        input[i].data[0]     = av_malloc(ysize + 2 * csize);

        input[i].data[1]     = input[i].data[0] + ysize;

        input[i].data[2]     = input[i].data[1] + csize;

        input[i].linesize[0] = c->width;

        input[i].linesize[1] =

        input[i].linesize[2] = c->width / 2;



        if (pre_input_ptr && (!i || s->input_picture[i - 1])) {

            pre_input = *pre_input_ptr;



            if (!pre_input.shared && i) {

                pre_input.f.data[0] += INPLACE_OFFSET;

                pre_input.f.data[1] += INPLACE_OFFSET;

                pre_input.f.data[2] += INPLACE_OFFSET;

            }



            s->dsp.shrink[scale](input[i].data[0], input[i].linesize[0],

                                 pre_input.f.data[0], pre_input.f.linesize[0],

                                 c->width,      c->height);

            s->dsp.shrink[scale](input[i].data[1], input[i].linesize[1],

                                 pre_input.f.data[1], pre_input.f.linesize[1],

                                 c->width >> 1, c->height >> 1);

            s->dsp.shrink[scale](input[i].data[2], input[i].linesize[2],

                                 pre_input.f.data[2], pre_input.f.linesize[2],

                                 c->width >> 1, c->height >> 1);

        }

    }



    for (j = 0; j < s->max_b_frames + 1; j++) {

        int64_t rd = 0;



        if (!s->input_picture[j])

            break;



        c->error[0] = c->error[1] = c->error[2] = 0;



        input[0].pict_type = AV_PICTURE_TYPE_I;

        input[0].quality   = 1 * FF_QP2LAMBDA;



        out_size = encode_frame(c, &input[0]);



        //rd += (out_size * lambda2) >> FF_LAMBDA_SHIFT;



        for (i = 0; i < s->max_b_frames + 1; i++) {

            int is_p = i % (j + 1) == j || i == s->max_b_frames;



            input[i + 1].pict_type = is_p ?

                                     AV_PICTURE_TYPE_P : AV_PICTURE_TYPE_B;

            input[i + 1].quality   = is_p ? p_lambda : b_lambda;



            out_size = encode_frame(c, &input[i + 1]);



            rd += (out_size * lambda2) >> (FF_LAMBDA_SHIFT - 3);

        }



        /* get the delayed frames */

        while (out_size) {

            out_size = encode_frame(c, NULL);

            rd += (out_size * lambda2) >> (FF_LAMBDA_SHIFT - 3);

        }



        rd += c->error[0] + c->error[1] + c->error[2];



        if (rd < best_rd) {

            best_rd = rd;

            best_b_count = j;

        }

    }



    avcodec_close(c);

    av_freep(&c);



    for (i = 0; i < s->max_b_frames + 2; i++) {

        av_freep(&input[i].data[0]);

    }



    return best_b_count;

}
