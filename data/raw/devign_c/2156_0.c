static int flashsv_encode_frame(AVCodecContext *avctx, uint8_t *buf, int buf_size, void *data)

{

    FlashSVContext * const s = avctx->priv_data;

    AVFrame *pict = data;

    AVFrame * const p = &s->frame;

    int res;

    int I_frame = 0;

    int opt_w, opt_h;



    *p = *pict;



    /* First frame needs to be a keyframe */

    if (avctx->frame_number == 0) {

        s->previous_frame = av_mallocz(p->linesize[0]*s->image_height);

        if (!s->previous_frame) {

            av_log(avctx, AV_LOG_ERROR, "Memory allocation failed.\n");

            return -1;

        }

        I_frame = 1;

    }



    /* Check the placement of keyframes */

    if (avctx->gop_size > 0) {

        if (avctx->frame_number >= s->last_key_frame + avctx->gop_size) {

            I_frame = 1;

        }

    }



#if 0

    int w, h;

    int optim_sizes[16][16];

    int smallest_size;

    //Try all possible combinations and store the encoded frame sizes

    for (w=1 ; w<17 ; w++) {

        for (h=1 ; h<17 ; h++) {

            optim_sizes[w-1][h-1] = encode_bitstream(s, p, s->encbuffer, s->image_width*s->image_height*4, w*16, h*16, s->previous_frame);

            //av_log(avctx, AV_LOG_ERROR, "[%d][%d]size = %d\n",w,h,optim_sizes[w-1][h-1]);

        }

    }



    //Search for the smallest framesize and encode the frame with those parameters

    smallest_size=optim_sizes[0][0];

    opt_w = 0;

    opt_h = 0;

    for (w=0 ; w<16 ; w++) {

        for (h=0 ; h<16 ; h++) {

            if (optim_sizes[w][h] < smallest_size) {

                smallest_size = optim_sizes[w][h];

                opt_w = w;

                opt_h = h;

            }

        }

    }

    res = encode_bitstream(s, p, buf, buf_size, (opt_w+1)*16, (opt_h+1)*16, s->previous_frame);

    av_log(avctx, AV_LOG_ERROR, "[%d][%d]optimal size = %d, res = %d|\n", opt_w, opt_h, smallest_size, res);



    if (buf_size < res)

        av_log(avctx, AV_LOG_ERROR, "buf_size %d < res %d\n", buf_size, res);



#else

    opt_w=1;

    opt_h=1;



    if (buf_size < s->image_width*s->image_height*3) {

        //Conservative upper bound check for compressed data

        av_log(avctx, AV_LOG_ERROR, "buf_size %d <  %d\n", buf_size, s->image_width*s->image_height*3);

        return -1;

    }



    res = encode_bitstream(s, p, buf, buf_size, opt_w*16, opt_h*16, s->previous_frame, &I_frame);

#endif

    //save the current frame

    memcpy(s->previous_frame, p->data[0], s->image_height*p->linesize[0]);



    //mark the frame type so the muxer can mux it correctly

    if (I_frame) {

        p->pict_type = FF_I_TYPE;

        p->key_frame = 1;

        s->last_key_frame = avctx->frame_number;

        av_log(avctx, AV_LOG_DEBUG, "Inserting key frame at frame %d\n",avctx->frame_number);

    } else {

        p->pict_type = FF_P_TYPE;

        p->key_frame = 0;

    }



    avctx->coded_frame = p;



    return res;

}
