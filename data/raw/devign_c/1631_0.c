static int h264_parse(AVCodecParserContext *s,

                      AVCodecContext *avctx,

                      uint8_t **poutbuf, int *poutbuf_size,

                      const uint8_t *buf, int buf_size)

{

    H264Context *h = s->priv_data;

    ParseContext *pc = &h->s.parse_context;

    int next;



    if(s->flags & PARSER_FLAG_COMPLETE_FRAMES){

        next= buf_size;

    }else{

        next= find_frame_end(h, buf, buf_size);



        if (ff_combine_frame(pc, next, (uint8_t **)&buf, &buf_size) < 0) {

            *poutbuf = NULL;

            *poutbuf_size = 0;

            return buf_size;

        }



        if(next<0){

            find_frame_end(h, &pc->buffer[pc->last_index + next], -next); //update state

        }

    }



    *poutbuf = (uint8_t *)buf;

    *poutbuf_size = buf_size;

    return next;

}
