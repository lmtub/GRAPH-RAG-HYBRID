AVFilterBufferRef *avfilter_default_get_video_buffer(AVFilterLink *link, int perms, int w, int h)

{

    AVFilterBuffer *pic = av_mallocz(sizeof(AVFilterBuffer));

    AVFilterBufferRef *ref = av_mallocz(sizeof(AVFilterBufferRef));

    int i, tempsize;

    char *buf;



    ref->buf         = pic;

    ref->video       = av_mallocz(sizeof(AVFilterBufferRefVideoProps));

    ref->video->w    = w;

    ref->video->h    = h;



    /* make sure the buffer gets read permission or it's useless for output */

    ref->perms = perms | AV_PERM_READ;



    pic->refcount = 1;

    ref->format   = link->format;

    pic->free     = avfilter_default_free_buffer;

    av_fill_image_linesizes(pic->linesize, ref->format, ref->video->w);



    for (i=0; i<4;i++)

        pic->linesize[i] = FFALIGN(pic->linesize[i], 16);



    tempsize = av_fill_image_pointers(pic->data, ref->format, ref->video->h, NULL, pic->linesize);

    buf = av_malloc(tempsize + 16); // +2 is needed for swscaler, +16 to be

                                    // SIMD-friendly

    av_fill_image_pointers(pic->data, ref->format, ref->video->h, buf, pic->linesize);



    memcpy(ref->data,     pic->data,     sizeof(ref->data));

    memcpy(ref->linesize, pic->linesize, sizeof(ref->linesize));



    return ref;

}
