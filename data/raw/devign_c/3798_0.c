static int video_get_buffer(AVCodecContext *s, AVFrame *pic)

{

    FramePool *pool = s->internal->pool;

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(pic->format);

    int i;



    if (pic->data[0]) {

        av_log(s, AV_LOG_ERROR, "pic->data[0]!=NULL in avcodec_default_get_buffer\n");

        return -1;

    }



    if (!desc) {

        av_log(s, AV_LOG_ERROR,

            "Unable to get pixel format descriptor for format %s\n",

            av_get_pix_fmt_name(pic->format));

        return AVERROR(EINVAL);

    }



    memset(pic->data, 0, sizeof(pic->data));

    pic->extended_data = pic->data;



    for (i = 0; i < 4 && pool->pools[i]; i++) {

        pic->linesize[i] = pool->linesize[i];



        pic->buf[i] = av_buffer_pool_get(pool->pools[i]);

        if (!pic->buf[i])

            goto fail;



        pic->data[i] = pic->buf[i]->data;

    }

    for (; i < AV_NUM_DATA_POINTERS; i++) {

        pic->data[i] = NULL;

        pic->linesize[i] = 0;

    }

    if (desc->flags & AV_PIX_FMT_FLAG_PAL ||

        desc->flags & AV_PIX_FMT_FLAG_PSEUDOPAL)

        avpriv_set_systematic_pal2((uint32_t *)pic->data[1], pic->format);



    if (s->debug & FF_DEBUG_BUFFERS)

        av_log(s, AV_LOG_DEBUG, "default_get_buffer called on pic %p\n", pic);



    return 0;

fail:

    av_frame_unref(pic);

    return AVERROR(ENOMEM);

}
