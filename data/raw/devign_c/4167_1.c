static int alloc_scratch_buffers(H264SliceContext *sl, int linesize)

{

    const H264Context *h = sl->h264;

    int alloc_size = FFALIGN(FFABS(linesize) + 32, 32);



    av_fast_malloc(&sl->bipred_scratchpad, &sl->bipred_scratchpad_allocated, 16 * 6 * alloc_size);

    // edge emu needs blocksize + filter length - 1

    // (= 21x21 for  h264)

    av_fast_malloc(&sl->edge_emu_buffer, &sl->edge_emu_buffer_allocated, alloc_size * 2 * 21);



    av_fast_malloc(&sl->top_borders[0], &sl->top_borders_allocated[0],

                   h->mb_width * 16 * 3 * sizeof(uint8_t) * 2);

    av_fast_malloc(&sl->top_borders[1], &sl->top_borders_allocated[1],

                   h->mb_width * 16 * 3 * sizeof(uint8_t) * 2);



    if (!sl->bipred_scratchpad || !sl->edge_emu_buffer ||

        !sl->top_borders[0]    || !sl->top_borders[1]) {

        av_freep(&sl->bipred_scratchpad);

        av_freep(&sl->edge_emu_buffer);

        av_freep(&sl->top_borders[0]);

        av_freep(&sl->top_borders[1]);



        sl->bipred_scratchpad_allocated = 0;

        sl->edge_emu_buffer_allocated   = 0;

        sl->top_borders_allocated[0]    = 0;

        sl->top_borders_allocated[1]    = 0;

        return AVERROR(ENOMEM);

    }



    return 0;

}
