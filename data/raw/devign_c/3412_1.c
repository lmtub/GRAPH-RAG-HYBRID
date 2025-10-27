static int applehttp_read_seek(AVFormatContext *s, int stream_index,

                               int64_t timestamp, int flags)

{

    AppleHTTPContext *c = s->priv_data;

    int i, j, ret;



    if ((flags & AVSEEK_FLAG_BYTE) || !c->variants[0]->finished)

        return AVERROR(ENOSYS);



    timestamp = av_rescale_rnd(timestamp, 1, stream_index >= 0 ?

                               s->streams[stream_index]->time_base.den :

                               AV_TIME_BASE, flags & AVSEEK_FLAG_BACKWARD ?

                               AV_ROUND_DOWN : AV_ROUND_UP);

    ret = AVERROR(EIO);

    for (i = 0; i < c->n_variants; i++) {

        /* Reset reading */

        struct variant *var = c->variants[i];

        int64_t pos = 0;

        if (var->input) {

            ffurl_close(var->input);

            var->input = NULL;

        }

        av_free_packet(&var->pkt);

        reset_packet(&var->pkt);

        var->pb.eof_reached = 0;



        /* Locate the segment that contains the target timestamp */

        for (j = 0; j < var->n_segments; j++) {

            if (timestamp >= pos &&

                timestamp < pos + var->segments[j]->duration) {

                var->cur_seq_no = var->start_seq_no + j;

                ret = 0;

                break;

            }

            pos += var->segments[j]->duration;

        }

    }

    return ret;

}
