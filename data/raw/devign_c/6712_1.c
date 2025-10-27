void ff_af_queue_remove(AudioFrameQueue *afq, int nb_samples, int64_t *pts,

                        int *duration)

{

    int64_t out_pts = AV_NOPTS_VALUE;

    int removed_samples = 0;

    int i;



    if (afq->frame_count || afq->frame_alloc) {

        if (afq->frames->pts != AV_NOPTS_VALUE)

            out_pts = afq->frames->pts;

    }

    if(!afq->frame_count)

        av_log(afq->avctx, AV_LOG_WARNING, "Trying to remove %d samples, but que empty\n", nb_samples);

    if (pts)

        *pts = ff_samples_to_time_base(afq->avctx, out_pts);



    for(i=0; nb_samples && i<afq->frame_count; i++){

        int n= FFMIN(afq->frames[i].duration, nb_samples);

        afq->frames[i].duration -= n;

        nb_samples              -= n;

        removed_samples         += n;

        if(afq->frames[i].pts != AV_NOPTS_VALUE)

            afq->frames[i].pts      += n;

    }

    i -= i && afq->frames[i-1].duration;

    memmove(afq->frames, afq->frames + i, sizeof(*afq->frames) * (afq->frame_count - i));

    afq->frame_count -= i;



    if(nb_samples){

        av_assert0(!afq->frame_count);

        if(afq->frames[0].pts != AV_NOPTS_VALUE)

            afq->frames[0].pts += nb_samples;

        av_log(afq->avctx, AV_LOG_DEBUG, "Trying to remove %d more samples than are in the que\n", nb_samples);

    }

    if (duration)

        *duration = ff_samples_to_time_base(afq->avctx, removed_samples);

}
