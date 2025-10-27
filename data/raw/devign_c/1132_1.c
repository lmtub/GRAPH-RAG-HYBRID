void av_cold ff_ivi_free_buffers(IVIPlaneDesc *planes)

{

    int p, b, t;



    for (p = 0; p < 3; p++) {

        for (b = 0; b < planes[p].num_bands; b++) {

            av_freep(&planes[p].bands[b].bufs[0]);

            av_freep(&planes[p].bands[b].bufs[1]);

            av_freep(&planes[p].bands[b].bufs[2]);





            for (t = 0; t < planes[p].bands[b].num_tiles; t++)

                av_freep(&planes[p].bands[b].tiles[t].mbs);

            av_freep(&planes[p].bands[b].tiles);

        }

        av_freep(&planes[p].bands);

    }

}