static int mpegaudio_parse(AVCodecParserContext *s1,

                           AVCodecContext *avctx,

                           uint8_t **poutbuf, int *poutbuf_size, 

                           const uint8_t *buf, int buf_size)

{

    MpegAudioParseContext *s = s1->priv_data;

    int len, ret;

    uint32_t header;

    const uint8_t *buf_ptr;



    *poutbuf = NULL;

    *poutbuf_size = 0;

    buf_ptr = buf;

    while (buf_size > 0) {

	len = s->inbuf_ptr - s->inbuf;

	if (s->frame_size == 0) {

            /* special case for next header for first frame in free

               format case (XXX: find a simpler method) */

            if (s->free_format_next_header != 0) {

                s->inbuf[0] = s->free_format_next_header >> 24;

                s->inbuf[1] = s->free_format_next_header >> 16;

                s->inbuf[2] = s->free_format_next_header >> 8;

                s->inbuf[3] = s->free_format_next_header;

                s->inbuf_ptr = s->inbuf + 4;

                s->free_format_next_header = 0;

                goto got_header;

            }

	    /* no header seen : find one. We need at least MPA_HEADER_SIZE

               bytes to parse it */

	    len = MPA_HEADER_SIZE - len;

	    if (len > buf_size)

		len = buf_size;

	    if (len > 0) {

		memcpy(s->inbuf_ptr, buf_ptr, len);

		buf_ptr += len;

		buf_size -= len;

		s->inbuf_ptr += len;

	    }

	    if ((s->inbuf_ptr - s->inbuf) >= MPA_HEADER_SIZE) {

            got_header:

		header = (s->inbuf[0] << 24) | (s->inbuf[1] << 16) |

		    (s->inbuf[2] << 8) | s->inbuf[3];



                ret = mpa_decode_header(avctx, header);

                if (ret < 0) {

		    /* no sync found : move by one byte (inefficient, but simple!) */

		    memmove(s->inbuf, s->inbuf + 1, s->inbuf_ptr - s->inbuf - 1);

		    s->inbuf_ptr--;

                    dprintf("skip %x\n", header);

                    /* reset free format frame size to give a chance

                       to get a new bitrate */

                    s->free_format_frame_size = 0;

		} else {

                    s->frame_size = ret;

#if 0

                    /* free format: prepare to compute frame size */

		    if (decode_header(s, header) == 1) {

			s->frame_size = -1;

                    }

#endif

		}

	    }

        } else 

#if 0

        if (s->frame_size == -1) {

            /* free format : find next sync to compute frame size */

	    len = MPA_MAX_CODED_FRAME_SIZE - len;

	    if (len > buf_size)

		len = buf_size;

            if (len == 0) {

		/* frame too long: resync */

                s->frame_size = 0;

		memmove(s->inbuf, s->inbuf + 1, s->inbuf_ptr - s->inbuf - 1);

		s->inbuf_ptr--;

            } else {

                uint8_t *p, *pend;

                uint32_t header1;

                int padding;



                memcpy(s->inbuf_ptr, buf_ptr, len);

                /* check for header */

                p = s->inbuf_ptr - 3;

                pend = s->inbuf_ptr + len - 4;

                while (p <= pend) {

                    header = (p[0] << 24) | (p[1] << 16) |

                        (p[2] << 8) | p[3];

                    header1 = (s->inbuf[0] << 24) | (s->inbuf[1] << 16) |

                        (s->inbuf[2] << 8) | s->inbuf[3];

                    /* check with high probability that we have a

                       valid header */

                    if ((header & SAME_HEADER_MASK) ==

                        (header1 & SAME_HEADER_MASK)) {

                        /* header found: update pointers */

                        len = (p + 4) - s->inbuf_ptr;

                        buf_ptr += len;

                        buf_size -= len;

                        s->inbuf_ptr = p;

                        /* compute frame size */

                        s->free_format_next_header = header;

                        s->free_format_frame_size = s->inbuf_ptr - s->inbuf;

                        padding = (header1 >> 9) & 1;

                        if (s->layer == 1)

                            s->free_format_frame_size -= padding * 4;

                        else

                            s->free_format_frame_size -= padding;

                        dprintf("free frame size=%d padding=%d\n", 

                                s->free_format_frame_size, padding);

                        decode_header(s, header1);

                        goto next_data;

                    }

                    p++;

                }

                /* not found: simply increase pointers */

                buf_ptr += len;

                s->inbuf_ptr += len;

                buf_size -= len;

            }

	} else 

#endif

        if (len < s->frame_size) {

            if (s->frame_size > MPA_MAX_CODED_FRAME_SIZE)

                s->frame_size = MPA_MAX_CODED_FRAME_SIZE;

	    len = s->frame_size - len;

	    if (len > buf_size)

		len = buf_size;

	    memcpy(s->inbuf_ptr, buf_ptr, len);

	    buf_ptr += len;

	    s->inbuf_ptr += len;

	    buf_size -= len;

	}

        //    next_data:

        if (s->frame_size > 0 && 

            (s->inbuf_ptr - s->inbuf) >= s->frame_size) {

            *poutbuf = s->inbuf;

            *poutbuf_size = s->inbuf_ptr - s->inbuf;

	    s->inbuf_ptr = s->inbuf;

	    s->frame_size = 0;

	    break;

	}

    }

    return buf_ptr - buf;

}
