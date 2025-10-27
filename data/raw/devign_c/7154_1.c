static av_cold int cinepak_encode_init(AVCodecContext *avctx)

{

    CinepakEncContext *s = avctx->priv_data;

    int x, mb_count, strip_buf_size, frame_buf_size;



    if (avctx->width & 3 || avctx->height & 3) {

        av_log(avctx, AV_LOG_ERROR, "width and height must be multiples of four (got %ix%i)\n",

                avctx->width, avctx->height);

        return AVERROR(EINVAL);

    }



    if (!(s->codebook_input = av_malloc(sizeof(int) * (avctx->pix_fmt == AV_PIX_FMT_YUV420P ? 6 : 4) * (avctx->width * avctx->height) >> 2)))

        return AVERROR(ENOMEM);



    if (!(s->codebook_closest = av_malloc(sizeof(int) * (avctx->width * avctx->height) >> 2)))

        goto enomem;



    for(x = 0; x < 3; x++)

        if(!(s->pict_bufs[x] = av_malloc((avctx->pix_fmt == AV_PIX_FMT_YUV420P ? 6 : 4) * (avctx->width * avctx->height) >> 2)))

            goto enomem;



    mb_count = avctx->width * avctx->height / MB_AREA;



    //the largest possible chunk is 0x31 with all MBs encoded in V4 mode, which is 34 bits per MB

    strip_buf_size = STRIP_HEADER_SIZE + 3 * CHUNK_HEADER_SIZE + 2 * VECTOR_MAX * CODEBOOK_MAX + 4 * (mb_count + (mb_count + 15) / 16);



    frame_buf_size = CVID_HEADER_SIZE + MAX_STRIPS * strip_buf_size;



    if (!(s->strip_buf = av_malloc(strip_buf_size)))

        goto enomem;



    if (!(s->frame_buf = av_malloc(frame_buf_size)))

        goto enomem;



    if (!(s->mb = av_malloc(mb_count*sizeof(mb_info))))

        goto enomem;



#ifdef CINEPAKENC_DEBUG

    if (!(s->best_mb = av_malloc(mb_count*sizeof(mb_info))))

        goto enomem;

#endif



    av_lfg_init(&s->randctx, 1);

    s->avctx = avctx;

    s->w = avctx->width;

    s->h = avctx->height;

    s->curframe = 0;

    s->keyint = avctx->keyint_min;

    s->pix_fmt = avctx->pix_fmt;



    //set up AVFrames

    s->last_frame.data[0]        = s->pict_bufs[0];

    s->last_frame.linesize[0]    = s->w;

    s->best_frame.data[0]        = s->pict_bufs[1];

    s->best_frame.linesize[0]    = s->w;

    s->scratch_frame.data[0]     = s->pict_bufs[2];

    s->scratch_frame.linesize[0] = s->w;



    if(s->pix_fmt == AV_PIX_FMT_YUV420P) {

        s->last_frame.data[1]        = s->last_frame.data[0] + s->w * s->h;

        s->last_frame.data[2]        = s->last_frame.data[1] + ((s->w * s->h) >> 2);

        s->last_frame.linesize[1]    = s->last_frame.linesize[2] = s->w >> 1;



        s->best_frame.data[1]        = s->best_frame.data[0] + s->w * s->h;

        s->best_frame.data[2]        = s->best_frame.data[1] + ((s->w * s->h) >> 2);

        s->best_frame.linesize[1]    = s->best_frame.linesize[2] = s->w >> 1;



        s->scratch_frame.data[1]     = s->scratch_frame.data[0] + s->w * s->h;

        s->scratch_frame.data[2]     = s->scratch_frame.data[1] + ((s->w * s->h) >> 2);

        s->scratch_frame.linesize[1] = s->scratch_frame.linesize[2] = s->w >> 1;

    }



    s->num_v1_mode = s->num_v4_mode = s->num_mc_mode = s->num_v1_encs = s->num_v4_encs = s->num_skips = 0;



    return 0;



enomem:

    av_free(s->codebook_input);

    av_free(s->codebook_closest);

    av_free(s->strip_buf);

    av_free(s->frame_buf);

    av_free(s->mb);

#ifdef CINEPAKENC_DEBUG

    av_free(s->best_mb);

#endif



    for(x = 0; x < 3; x++)

        av_free(s->pict_bufs[x]);



    return AVERROR(ENOMEM);

}
