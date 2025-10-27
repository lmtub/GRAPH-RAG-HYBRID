static int input_get_buffer(AVCodecContext *codec, AVFrame *pic)

{

    AVFilterContext *ctx = codec->opaque;

    AVFilterBufferRef  *ref;

    int perms = AV_PERM_WRITE;

    int i, w, h, stride[4];

    unsigned edge;



    if(av_image_check_size(w, h, 0, codec))

        return -1;



    if (codec->codec->capabilities & CODEC_CAP_NEG_LINESIZES)

        perms |= AV_PERM_NEG_LINESIZES;



    if(pic->buffer_hints & FF_BUFFER_HINTS_VALID) {

        if(pic->buffer_hints & FF_BUFFER_HINTS_READABLE) perms |= AV_PERM_READ;

        if(pic->buffer_hints & FF_BUFFER_HINTS_PRESERVE) perms |= AV_PERM_PRESERVE;

        if(pic->buffer_hints & FF_BUFFER_HINTS_REUSABLE) perms |= AV_PERM_REUSE2;

    }

    if(pic->reference) perms |= AV_PERM_READ | AV_PERM_PRESERVE;



    w = codec->width;

    h = codec->height;

    avcodec_align_dimensions2(codec, &w, &h, stride);

    edge = codec->flags & CODEC_FLAG_EMU_EDGE ? 0 : avcodec_get_edge_width();

    w += edge << 1;

    h += edge << 1;



    if(!(ref = avfilter_get_video_buffer(ctx->outputs[0], perms, w, h)))

        return -1;



    ref->video->w = codec->width;

    ref->video->h = codec->height;

    for(i = 0; i < 4; i ++) {

        unsigned hshift = (i == 1 || i == 2) ? av_pix_fmt_descriptors[ref->format].log2_chroma_w : 0;

        unsigned vshift = (i == 1 || i == 2) ? av_pix_fmt_descriptors[ref->format].log2_chroma_h : 0;



        if (ref->data[i]) {

            ref->data[i]    += (edge >> hshift) + ((edge * ref->linesize[i]) >> vshift);

        }

        pic->data[i]     = ref->data[i];

        pic->linesize[i] = ref->linesize[i];

    }

    pic->opaque = ref;

    pic->age    = INT_MAX;

    pic->type   = FF_BUFFER_TYPE_USER;

    pic->reordered_opaque = codec->reordered_opaque;

    if(codec->pkt) pic->pkt_pts = codec->pkt->pts;

    else           pic->pkt_pts = AV_NOPTS_VALUE;

    return 0;

}
