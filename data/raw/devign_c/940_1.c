void ff_dv_offset_reset(DVDemuxContext *c, int64_t frame_offset)

{

    c->frames= frame_offset;

    if (c->ach)

        c->abytes= av_rescale_q(c->frames, c->sys->time_base,

                                (AVRational){8, c->ast[0]->codec->bit_rate});

    c->audio_pkt[0].size = c->audio_pkt[1].size = 0;

    c->audio_pkt[2].size = c->audio_pkt[3].size = 0;

}
