void ff_framequeue_skip_samples(FFFrameQueue *fq, size_t samples, AVRational time_base)

{

    FFFrameBucket *b;

    size_t bytes;

    int planar, planes, i;



    check_consistency(fq);

    av_assert1(fq->queued);

    b = bucket(fq, 0);

    av_assert1(samples < b->frame->nb_samples);

    planar = av_sample_fmt_is_planar(b->frame->format);

    planes = planar ? b->frame->channels : 1;

    bytes = samples * av_get_bytes_per_sample(b->frame->format);

    if (!planar)

        bytes *= b->frame->channels;

    if (b->frame->pts != AV_NOPTS_VALUE)

        b->frame->pts += av_rescale_q(samples, av_make_q(1, b->frame->sample_rate), time_base);

    b->frame->nb_samples -= samples;

    b->frame->linesize[0] -= bytes;

    for (i = 0; i < planes; i++)

        b->frame->extended_data[i] += bytes;

    for (i = 0; i < planes && i < AV_NUM_DATA_POINTERS; i++)

        b->frame->data[i] = b->frame->extended_data[i];

    fq->total_samples_tail += samples;


    ff_framequeue_update_peeked(fq, 0);

}