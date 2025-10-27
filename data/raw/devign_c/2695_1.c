static void reset_codec(WmallDecodeCtx *s)

{

    int ich, ilms;

    s->mclms_recent = s->mclms_order * s->num_channels;

    for (ich = 0; ich < s->num_channels; ich++) {

        for (ilms = 0; ilms < s->cdlms_ttl[ich]; ilms++)

            s->cdlms[ich][ilms].recent = s->cdlms[ich][ilms].order;

        /* first sample of a seekable subframe is considered as the starting of

           a transient area which is samples_per_frame samples long */

        s->channel[ich].transient_counter = s->samples_per_frame;

        s->transient[ich] = 1;

    }

}
