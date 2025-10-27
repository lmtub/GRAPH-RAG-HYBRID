static int alloc_sequence_buffers(DiracContext *s)

{

    int sbwidth  = DIVRNDUP(s->source.width,  4);

    int sbheight = DIVRNDUP(s->source.height, 4);

    int i, w, h, top_padding;



    /* todo: think more about this / use or set Plane here */

    for (i = 0; i < 3; i++) {

        int max_xblen = MAX_BLOCKSIZE >> (i ? s->chroma_x_shift : 0);

        int max_yblen = MAX_BLOCKSIZE >> (i ? s->chroma_y_shift : 0);

        w = s->source.width  >> (i ? s->chroma_x_shift : 0);

        h = s->source.height >> (i ? s->chroma_y_shift : 0);



        /* we allocate the max we support here since num decompositions can

         * change from frame to frame. Stride is aligned to 16 for SIMD, and

         * 1<<MAX_DWT_LEVELS top padding to avoid if(y>0) in arith decoding

         * MAX_BLOCKSIZE padding for MC: blocks can spill up to half of that

         * on each side */

        top_padding = FFMAX(1<<MAX_DWT_LEVELS, max_yblen/2);

        w = FFALIGN(CALC_PADDING(w, MAX_DWT_LEVELS), 8); /* FIXME: Should this be 16 for SSE??? */

        h = top_padding + CALC_PADDING(h, MAX_DWT_LEVELS) + max_yblen/2;



        s->plane[i].idwt_buf_base = av_mallocz((w+max_xblen)*h * sizeof(IDWTELEM));

        s->plane[i].idwt_tmp      = av_malloc((w+16) * sizeof(IDWTELEM));

        s->plane[i].idwt_buf      = s->plane[i].idwt_buf_base + top_padding*w;

        if (!s->plane[i].idwt_buf_base || !s->plane[i].idwt_tmp)

            return AVERROR(ENOMEM);

    }



    w = s->source.width;

    h = s->source.height;



    /* fixme: allocate using real stride here */

    s->sbsplit  = av_malloc(sbwidth * sbheight);

    s->blmotion = av_malloc(sbwidth * sbheight * 16 * sizeof(*s->blmotion));

    s->edge_emu_buffer_base = av_malloc((w+64)*MAX_BLOCKSIZE);



    s->mctmp     = av_malloc((w+64+MAX_BLOCKSIZE) * (h*MAX_BLOCKSIZE) * sizeof(*s->mctmp));

    s->mcscratch = av_malloc((w+64)*MAX_BLOCKSIZE);



    if (!s->sbsplit || !s->blmotion)

        return AVERROR(ENOMEM);

    return 0;

}
