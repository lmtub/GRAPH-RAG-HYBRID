int avcodec_default_reget_buffer(AVCodecContext *s, AVFrame *pic){

    AVFrame temp_pic;

    int i;



    assert(s->codec_type == AVMEDIA_TYPE_VIDEO);



    if (pic->data[0] && (pic->width != s->width || pic->height != s->height || pic->format != s->pix_fmt)) {

        av_log(s, AV_LOG_WARNING, "Picture changed from size:%dx%d fmt:%s to size:%dx%d fmt:%s in reget buffer()\n",

               pic->width, pic->height, av_get_pix_fmt_name(pic->format), s->width, s->height, av_get_pix_fmt_name(s->pix_fmt));

        s->release_buffer(s, pic);

    }



    /* If no picture return a new buffer */

    if(pic->data[0] == NULL) {

        /* We will copy from buffer, so must be readable */

        pic->buffer_hints |= FF_BUFFER_HINTS_READABLE;

        return s->get_buffer(s, pic);

    }



    /* If internal buffer type return the same buffer */

    if(pic->type == FF_BUFFER_TYPE_INTERNAL) {

        if(s->pkt) pic->pkt_pts= s->pkt->pts;

        else       pic->pkt_pts= AV_NOPTS_VALUE;

        pic->reordered_opaque= s->reordered_opaque;

        return 0;

    }



    /*

     * Not internal type and reget_buffer not overridden, emulate cr buffer

     */

    temp_pic = *pic;

    for(i = 0; i < AV_NUM_DATA_POINTERS; i++)

        pic->data[i] = pic->base[i] = NULL;

    pic->opaque = NULL;

    /* Allocate new frame */

    if (s->get_buffer(s, pic))

        return -1;

    /* Copy image data from old buffer to new buffer */

    av_picture_copy((AVPicture*)pic, (AVPicture*)&temp_pic, s->pix_fmt, s->width,

             s->height);

    s->release_buffer(s, &temp_pic); // Release old frame

    return 0;

}
