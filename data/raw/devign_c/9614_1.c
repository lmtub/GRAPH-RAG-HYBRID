struct SwrContext *swr_alloc_set_opts(struct SwrContext *s,

                                      int64_t out_ch_layout, enum AVSampleFormat out_sample_fmt, int out_sample_rate,

                                      int64_t  in_ch_layout, enum AVSampleFormat  in_sample_fmt, int  in_sample_rate,

                                      int log_offset, void *log_ctx){

    if(!s) s= swr_alloc();

    if(!s) return NULL;



    s->log_level_offset= log_offset;

    s->log_ctx= log_ctx;



    av_opt_set_int(s, "ocl", out_ch_layout,   0);

    av_opt_set_int(s, "osf", out_sample_fmt,  0);

    av_opt_set_int(s, "osr", out_sample_rate, 0);

    av_opt_set_int(s, "icl", in_ch_layout,    0);

    av_opt_set_int(s, "isf", in_sample_fmt,   0);

    av_opt_set_int(s, "isr", in_sample_rate,  0);

    av_opt_set_int(s, "tsf", AV_SAMPLE_FMT_NONE,   0);

    av_opt_set_int(s, "ich", av_get_channel_layout_nb_channels(s-> in_ch_layout), 0);

    av_opt_set_int(s, "och", av_get_channel_layout_nb_channels(s->out_ch_layout), 0);

    av_opt_set_int(s, "uch", 0, 0);

    return s;

}
