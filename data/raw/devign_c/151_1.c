static int flac_parse(AVCodecParserContext *s, AVCodecContext *avctx,

                      const uint8_t **poutbuf, int *poutbuf_size,

                      const uint8_t *buf, int buf_size)

{

    FLACParseContext *fpc = s->priv_data;

    FLACHeaderMarker *curr;

    int nb_headers;

    const uint8_t *read_end   = buf;

    const uint8_t *read_start = buf;



    if (s->flags & PARSER_FLAG_COMPLETE_FRAMES) {

        FLACFrameInfo fi;

        if (frame_header_is_valid(avctx, buf, &fi)) {

            s->duration = fi.blocksize;

            if (!avctx->sample_rate)

                avctx->sample_rate = fi.samplerate;

            if (fpc->pc->flags & PARSER_FLAG_USE_CODEC_TS){

                fpc->pc->pts = fi.frame_or_sample_num;

                if (!fi.is_var_size)

                  fpc->pc->pts *= fi.blocksize;

            }

        }

        *poutbuf      = buf;

        *poutbuf_size = buf_size;

        return buf_size;

    }



    fpc->avctx = avctx;

    if (fpc->best_header_valid)

        return get_best_header(fpc, poutbuf, poutbuf_size);



    /* If a best_header was found last call remove it with the buffer data. */

    if (fpc->best_header && fpc->best_header->best_child) {

        FLACHeaderMarker *temp;

        FLACHeaderMarker *best_child = fpc->best_header->best_child;



        /* Remove headers in list until the end of the best_header. */

        for (curr = fpc->headers; curr != best_child; curr = temp) {

            if (curr != fpc->best_header) {

                av_log(avctx, AV_LOG_DEBUG,

                       "dropping low score %i frame header from offset %i to %i\n",

                       curr->max_score, curr->offset, curr->next->offset);

            }

            temp = curr->next;

            av_freep(&curr->link_penalty);

            av_free(curr);


        }

        /* Release returned data from ring buffer. */

        av_fifo_drain(fpc->fifo_buf, best_child->offset);



        /* Fix the offset for the headers remaining to match the new buffer. */

        for (curr = best_child->next; curr; curr = curr->next)

            curr->offset -= best_child->offset;




        best_child->offset = 0;

        fpc->headers       = best_child;

        if (fpc->nb_headers_buffered >= FLAC_MIN_HEADERS) {

            fpc->best_header = best_child;

            return get_best_header(fpc, poutbuf, poutbuf_size);

        }

        fpc->best_header   = NULL;

    } else if (fpc->best_header) {

        /* No end frame no need to delete the buffer; probably eof */

        FLACHeaderMarker *temp;



        for (curr = fpc->headers; curr != fpc->best_header; curr = temp) {

            temp = curr->next;

            av_freep(&curr->link_penalty);

            av_free(curr);


        }

        fpc->headers = fpc->best_header->next;

        av_freep(&fpc->best_header->link_penalty);

        av_freep(&fpc->best_header);


    }



    /* Find and score new headers.                                     */

    /* buf_size is to zero when padding, so check for this since we do */

    /* not want to try to read more input once we have found the end.  */

    /* Note that as (non-modified) parameters, buf can be non-NULL,    */

    /* while buf_size is 0.                                            */

    while ((buf && buf_size && read_end < buf + buf_size &&

            fpc->nb_headers_buffered < FLAC_MIN_HEADERS)

           || ((!buf || !buf_size) && !fpc->end_padded)) {

        int start_offset;



        /* Pad the end once if EOF, to check the final region for headers. */

        if (!buf || !buf_size) {

            fpc->end_padded      = 1;

            buf_size = MAX_FRAME_HEADER_SIZE;

            read_end = read_start + MAX_FRAME_HEADER_SIZE;

        } else {

            /* The maximum read size is the upper-bound of what the parser

               needs to have the required number of frames buffered */

            int nb_desired = FLAC_MIN_HEADERS - fpc->nb_headers_buffered + 1;

            read_end       = read_end + FFMIN(buf + buf_size - read_end,

                                              nb_desired * FLAC_AVG_FRAME_SIZE);

        }



        if (!av_fifo_space(fpc->fifo_buf) &&

            av_fifo_size(fpc->fifo_buf) / FLAC_AVG_FRAME_SIZE >

            fpc->nb_headers_buffered * 20) {

            /* There is less than one valid flac header buffered for 20 headers

             * buffered. Therefore the fifo is most likely filled with invalid

             * data and the input is not a flac file. */

            goto handle_error;

        }



        /* Fill the buffer. */

        if (   av_fifo_space(fpc->fifo_buf) < read_end - read_start

            && av_fifo_realloc2(fpc->fifo_buf, (read_end - read_start) + 2*av_fifo_size(fpc->fifo_buf)) < 0) {

            av_log(avctx, AV_LOG_ERROR,

                   "couldn't reallocate buffer of size %"PTRDIFF_SPECIFIER"\n",

                   (read_end - read_start) + av_fifo_size(fpc->fifo_buf));

            goto handle_error;

        }



        if (buf && buf_size) {

            av_fifo_generic_write(fpc->fifo_buf, (void*) read_start,

                                  read_end - read_start, NULL);

        } else {

            int8_t pad[MAX_FRAME_HEADER_SIZE] = { 0 };

            av_fifo_generic_write(fpc->fifo_buf, (void*) pad, sizeof(pad), NULL);

        }



        /* Tag headers and update sequences. */

        start_offset = av_fifo_size(fpc->fifo_buf) -

                       ((read_end - read_start) + (MAX_FRAME_HEADER_SIZE - 1));

        start_offset = FFMAX(0, start_offset);

        nb_headers   = find_new_headers(fpc, start_offset);



        if (nb_headers < 0) {

            av_log(avctx, AV_LOG_ERROR,

                   "find_new_headers couldn't allocate FLAC header\n");

            goto handle_error;

        }



        fpc->nb_headers_buffered = nb_headers;

        /* Wait till FLAC_MIN_HEADERS to output a valid frame. */

        if (!fpc->end_padded && fpc->nb_headers_buffered < FLAC_MIN_HEADERS) {

            if (buf && read_end < buf + buf_size) {

                read_start = read_end;

                continue;

            } else {

                goto handle_error;

            }

        }



        /* If headers found, update the scores since we have longer chains. */

        if (fpc->end_padded || fpc->nb_headers_found)

            score_sequences(fpc);



        /* restore the state pre-padding */

        if (fpc->end_padded) {

            int warp = fpc->fifo_buf->wptr - fpc->fifo_buf->buffer < MAX_FRAME_HEADER_SIZE;

            /* HACK: drain the tail of the fifo */

            fpc->fifo_buf->wptr -= MAX_FRAME_HEADER_SIZE;

            fpc->fifo_buf->wndx -= MAX_FRAME_HEADER_SIZE;

            if (warp) {

                fpc->fifo_buf->wptr += fpc->fifo_buf->end -

                    fpc->fifo_buf->buffer;

            }

            buf_size = 0;

            read_start = read_end = NULL;

        }

    }



    for (curr = fpc->headers; curr; curr = curr->next) {

        if (curr->max_score > 0 &&

            (!fpc->best_header || curr->max_score > fpc->best_header->max_score)) {

            fpc->best_header = curr;

        }

    }



    if (fpc->best_header) {

        fpc->best_header_valid = 1;

        if (fpc->best_header->offset > 0) {

            /* Output a junk frame. */

            av_log(avctx, AV_LOG_DEBUG, "Junk frame till offset %i\n",

                   fpc->best_header->offset);



            /* Set duration to 0. It is unknown or invalid in a junk frame. */

            s->duration = 0;

            *poutbuf_size     = fpc->best_header->offset;

            *poutbuf          = flac_fifo_read_wrap(fpc, 0, *poutbuf_size,

                                                    &fpc->wrap_buf,

                                                    &fpc->wrap_buf_allocated_size);

            return buf_size ? (read_end - buf) : (fpc->best_header->offset -

                                           av_fifo_size(fpc->fifo_buf));

        }

        if (!buf_size)

            return get_best_header(fpc, poutbuf, poutbuf_size);

    }



handle_error:

    *poutbuf      = NULL;

    *poutbuf_size = 0;

    return buf_size ? read_end - buf : 0;

}