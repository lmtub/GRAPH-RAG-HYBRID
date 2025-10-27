void ff_fetch_timestamp(AVCodecParserContext *s, int off, int remove)

{

    int i;



    s->dts    =

    s->pts    = AV_NOPTS_VALUE;

    s->pos    = -1;

    s->offset = 0;

    for (i = 0; i < AV_PARSER_PTS_NB; i++) {

        if (s->cur_offset + off >= s->cur_frame_offset[i] &&

            (s->frame_offset < s->cur_frame_offset[i] ||

             (!s->frame_offset && !s->next_frame_offset)) && // first field/frame

            // check disabled since MPEG-TS does not send complete PES packets

            /*s->next_frame_offset + off <*/  s->cur_frame_end[i]){



            s->dts    = s->cur_frame_dts[i];

            s->pts    = s->cur_frame_pts[i];

            s->pos    = s->cur_frame_pos[i];

            s->offset = s->next_frame_offset - s->cur_frame_offset[i];

            if (remove)

                s->cur_frame_offset[i] = INT64_MAX;

            if (s->cur_offset + off < s->cur_frame_end[i])

                break;

        }

    }

}
