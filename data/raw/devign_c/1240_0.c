int ff_af_queue_add(AudioFrameQueue *afq, const AVFrame *f)

{

    AudioFrame *new_frame;

    AudioFrame *queue_end = afq->frame_queue;



    /* find the end of the queue */

    while (queue_end && queue_end->next)

        queue_end = queue_end->next;



    /* allocate new frame queue entry */

    if (!(new_frame = av_malloc(sizeof(*new_frame))))

        return AVERROR(ENOMEM);



    /* get frame parameters */

    new_frame->next = NULL;

    new_frame->duration = f->nb_samples;

    if (f->pts != AV_NOPTS_VALUE) {

        new_frame->pts = av_rescale_q(f->pts,

                                      afq->avctx->time_base,

                                      (AVRational){ 1, afq->avctx->sample_rate });

        afq->next_pts = new_frame->pts + new_frame->duration;

    } else {

        new_frame->pts = AV_NOPTS_VALUE;

        afq->next_pts  = AV_NOPTS_VALUE;

    }



    /* add new frame to the end of the queue */

    if (!queue_end)

        afq->frame_queue = new_frame;

    else

        queue_end->next = new_frame;



    /* add frame sample count */

    afq->remaining_samples += f->nb_samples;



#ifdef DEBUG

    ff_af_queue_log_state(afq);

#endif



    return 0;

}
