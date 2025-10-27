int ff_audio_interleave_init(AVFormatContext *s,

                             const int *samples_per_frame,

                             AVRational time_base)

{

    int i;



    if (!samples_per_frame)

        return -1;



    if (!time_base.num) {

        av_log(s, AV_LOG_ERROR, "timebase not set for audio interleave\n");

        return -1;

    }

    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        AudioInterleaveContext *aic = st->priv_data;



        if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            aic->sample_size = (st->codec->channels *

                                av_get_bits_per_sample(st->codec->codec_id)) / 8;

            if (!aic->sample_size) {

                av_log(s, AV_LOG_ERROR, "could not compute sample size\n");

                return -1;

            }

            aic->samples_per_frame = samples_per_frame;

            aic->samples = aic->samples_per_frame;

            aic->time_base = time_base;



            aic->fifo_size = 100* *aic->samples;

            aic->fifo= av_fifo_alloc_array(100, *aic->samples);

        }

    }



    return 0;

}
