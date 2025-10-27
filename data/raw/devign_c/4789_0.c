static void dump_cook_context(COOKContext *q, COOKextradata *e)

{

    //int i=0;

#define PRINT(a,b) av_log(NULL,AV_LOG_ERROR," %s = %d\n", a, b);

    av_log(NULL,AV_LOG_ERROR,"COOKextradata\n");

    av_log(NULL,AV_LOG_ERROR,"cookversion=%x\n",e->cookversion);

    if (e->cookversion > MONO_COOK2) {

        PRINT("js_subband_start",e->js_subband_start);

        PRINT("js_vlc_bits",e->js_vlc_bits);

    }

    av_log(NULL,AV_LOG_ERROR,"COOKContext\n");

    PRINT("nb_channels",q->nb_channels);

    PRINT("bit_rate",q->bit_rate);

    PRINT("sample_rate",q->sample_rate);

    PRINT("samples_per_channel",q->samples_per_channel);

    PRINT("samples_per_frame",q->samples_per_frame);

    PRINT("subbands",q->subbands);

    PRINT("random_state",q->random_state);

    PRINT("mlt_size",q->mlt_size);

    PRINT("js_subband_start",q->js_subband_start);

    PRINT("numvector_bits",q->numvector_bits);

    PRINT("numvector_size",q->numvector_size);

    PRINT("total_subbands",q->total_subbands);

    PRINT("frame_reorder_counter",q->frame_reorder_counter);

    PRINT("frame_reorder_index_size",q->frame_reorder_index_size);

}
