static int vc1_parse(AVCodecParserContext *s,

                           AVCodecContext *avctx,

                           const uint8_t **poutbuf, int *poutbuf_size,

                           const uint8_t *buf, int buf_size)

{

    /* Here we do the searching for frame boundaries and headers at

     * the same time. Only a minimal amount at the start of each

     * header is unescaped. */

    VC1ParseContext *vpc = s->priv_data;

    int pic_found = vpc->pc.frame_start_found;

    uint8_t *unesc_buffer = vpc->unesc_buffer;

    size_t unesc_index = vpc->unesc_index;

    VC1ParseSearchState search_state = vpc->search_state;

    int start_code_found;

    int next = END_NOT_FOUND;

    int i = vpc->bytes_to_skip;



    if (pic_found && buf_size == 0) {

        /* EOF considered as end of frame */

        memset(unesc_buffer + unesc_index, 0, UNESCAPED_THRESHOLD - unesc_index);

        vc1_extract_header(s, avctx, unesc_buffer, unesc_index);

        next = 0;

    }

    while (i < buf_size) {

        uint8_t b;

        start_code_found = 0;

        while (i < buf_size && unesc_index < UNESCAPED_THRESHOLD) {

            b = buf[i++];

            unesc_buffer[unesc_index++] = b;

            if (search_state <= ONE_ZERO)

                search_state = b ? NO_MATCH : search_state + 1;

            else if (search_state == TWO_ZEROS) {

                if (b == 1)

                    search_state = ONE;

                else if (b > 1) {

                    if (b == 3)

                        unesc_index--; // swallow emulation prevention byte

                    search_state = NO_MATCH;

                }

            }

            else { // search_state == ONE

                // Header unescaping terminates early due to detection of next start code

                search_state = NO_MATCH;

                start_code_found = 1;

                break;

            }

        }

        if ((s->flags & PARSER_FLAG_COMPLETE_FRAMES) &&

                unesc_index >= UNESCAPED_THRESHOLD &&

                vpc->prev_start_code == (VC1_CODE_FRAME & 0xFF))

        {

            // No need to keep scanning the rest of the buffer for

            // start codes if we know it contains a complete frame and

            // we've already unescaped all we need of the frame header

            vc1_extract_header(s, avctx, unesc_buffer, unesc_index);

            break;

        }

        if (unesc_index >= UNESCAPED_THRESHOLD && !start_code_found) {

            while (i < buf_size) {

                if (search_state == NO_MATCH) {

                    i += vpc->v.vc1dsp.startcode_find_candidate(buf + i, buf_size - i);

                    if (i < buf_size) {

                        search_state = ONE_ZERO;

                    }

                    i++;

                } else {

                    b = buf[i++];

                    if (search_state == ONE_ZERO)

                        search_state = b ? NO_MATCH : TWO_ZEROS;

                    else if (search_state == TWO_ZEROS) {

                        if (b >= 1)

                            search_state = b == 1 ? ONE : NO_MATCH;

                    }

                    else { // search_state == ONE

                        search_state = NO_MATCH;

                        start_code_found = 1;

                        break;

                    }

                }

            }

        }

        if (start_code_found) {

            vc1_extract_header(s, avctx, unesc_buffer, unesc_index);



            vpc->prev_start_code = b;

            unesc_index = 0;



            if (!(s->flags & PARSER_FLAG_COMPLETE_FRAMES)) {

                if (!pic_found && (b == (VC1_CODE_FRAME & 0xFF) || b == (VC1_CODE_FIELD & 0xFF))) {

                    pic_found = 1;

                }

                else if (pic_found && b != (VC1_CODE_FIELD & 0xFF) && b != (VC1_CODE_SLICE & 0xFF)) {

                    next = i - 4;

                    pic_found = b == (VC1_CODE_FRAME & 0xFF);

                    break;

                }

            }

        }

    }



    vpc->pc.frame_start_found = pic_found;

    vpc->unesc_index = unesc_index;

    vpc->search_state = search_state;



    if (s->flags & PARSER_FLAG_COMPLETE_FRAMES) {

        next = buf_size;

    } else {

        if (ff_combine_frame(&vpc->pc, next, &buf, &buf_size) < 0) {

            vpc->bytes_to_skip = 0;

            *poutbuf = NULL;

            *poutbuf_size = 0;

            return buf_size;

        }

    }



    /* If we return with a valid pointer to a combined frame buffer

     * then on the next call then we'll have been unhelpfully rewound

     * by up to 4 bytes (depending upon whether the start code

     * overlapped the input buffer, and if so by how much). We don't

     * want this: it will either cause spurious second detections of

     * the start code we've already seen, or cause extra bytes to be

     * inserted at the start of the unescaped buffer. */

    vpc->bytes_to_skip = 4;

    if (next < 0 && start_code_found)

        vpc->bytes_to_skip += next;



    *poutbuf = buf;

    *poutbuf_size = buf_size;

    return next;

}
