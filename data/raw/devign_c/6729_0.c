int ff_interleave_add_packet(AVFormatContext *s, AVPacket *pkt,

                              int (*compare)(AVFormatContext *, AVPacket *, AVPacket *))

{

    AVPacketList **next_point, *this_pktl;

    AVStream *st   = s->streams[pkt->stream_index];

    int chunked    = s->max_chunk_size || s->max_chunk_duration;



    this_pktl      = av_mallocz(sizeof(AVPacketList));

    if (!this_pktl)

        return AVERROR(ENOMEM);

    this_pktl->pkt = *pkt;

    pkt->destruct  = NULL;           // do not free original but only the copy

    av_dup_packet(&this_pktl->pkt);  // duplicate the packet if it uses non-allocated memory



    if (s->streams[pkt->stream_index]->last_in_packet_buffer) {

        next_point = &(st->last_in_packet_buffer->next);

    } else {

        next_point = &s->packet_buffer;

    }



    if (chunked) {

        uint64_t max= av_rescale_q_rnd(s->max_chunk_duration, AV_TIME_BASE_Q, st->time_base, AV_ROUND_UP);

        st->interleaver_chunk_size     += pkt->size;

        st->interleaver_chunk_duration += pkt->duration;

        if (   st->interleaver_chunk_size     > s->max_chunk_size-1U

            || st->interleaver_chunk_duration > max-1U) {

            st->interleaver_chunk_size     =

            st->interleaver_chunk_duration = 0;

            this_pktl->pkt.flags |= CHUNK_START;

        }

    }

    if (*next_point) {

        if (chunked && !(this_pktl->pkt.flags & CHUNK_START))

            goto next_non_null;



        if (compare(s, &s->packet_buffer_end->pkt, pkt)) {

            while (   *next_point

                   && ((chunked && !((*next_point)->pkt.flags&CHUNK_START))

                       || !compare(s, &(*next_point)->pkt, pkt)))

                next_point = &(*next_point)->next;

            if (*next_point)

                goto next_non_null;

        } else {

            next_point = &(s->packet_buffer_end->next);

        }

    }

    av_assert1(!*next_point);



    s->packet_buffer_end = this_pktl;

next_non_null:



    this_pktl->next = *next_point;



    s->streams[pkt->stream_index]->last_in_packet_buffer =

        *next_point                                      = this_pktl;

    return 0;

}
