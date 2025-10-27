static int decode_frame(AVCodecContext * avctx,

			void *data, int *data_size,

			uint8_t * buf, int buf_size)

{

    MPADecodeContext *s = avctx->priv_data;

    uint32_t header;

    uint8_t *buf_ptr;

    int len, out_size;

    OUT_INT *out_samples = data;



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

	    /* no header seen : find one. We need at least HEADER_SIZE

               bytes to parse it */

	    len = HEADER_SIZE - len;

	    if (len > buf_size)

		len = buf_size;

	    if (len > 0) {

		memcpy(s->inbuf_ptr, buf_ptr, len);

		buf_ptr += len;

		buf_size -= len;

		s->inbuf_ptr += len;

	    }

	    if ((s->inbuf_ptr - s->inbuf) >= HEADER_SIZE) {

            got_header:

		header = (s->inbuf[0] << 24) | (s->inbuf[1] << 16) |

		    (s->inbuf[2] << 8) | s->inbuf[3];



		if (ff_mpa_check_header(header) < 0) {

		    /* no sync found : move by one byte (inefficient, but simple!) */

		    memmove(s->inbuf, s->inbuf + 1, s->inbuf_ptr - s->inbuf - 1);

		    s->inbuf_ptr--;

                    dprintf("skip %x\n", header);

                    /* reset free format frame size to give a chance

                       to get a new bitrate */

                    s->free_format_frame_size = 0;

		} else {

		    if (decode_header(s, header) == 1) {

                        /* free format: prepare to compute frame size */

			s->frame_size = -1;

                    }

                    /* update codec info */

                    avctx->sample_rate = s->sample_rate;

                    avctx->channels = s->nb_channels;

                    avctx->bit_rate = s->bit_rate;

                    avctx->sub_id = s->layer;

                    switch(s->layer) {

                    case 1:

                        avctx->frame_size = 384;

                        break;

                    case 2:

                        avctx->frame_size = 1152;

                        break;

                    case 3:

                        if (s->lsf)

                            avctx->frame_size = 576;

                        else

                            avctx->frame_size = 1152;

                        break;

                    }

		}

	    }

        } else if (s->frame_size == -1) {

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

	} else if (len < s->frame_size) {

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

    next_data:

        if (s->frame_size > 0 && 

            (s->inbuf_ptr - s->inbuf) >= s->frame_size) {

            if (avctx->parse_only) {

                /* simply return the frame data */

                *(uint8_t **)data = s->inbuf;

                out_size = s->inbuf_ptr - s->inbuf;

            } else {

                out_size = mp_decode_frame(s, out_samples);

            }

	    s->inbuf_ptr = s->inbuf;

	    s->frame_size = 0;

	    *data_size = out_size;

	    break;

	}

    }

    return buf_ptr - buf;

}
