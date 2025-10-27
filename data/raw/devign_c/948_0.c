static int mjpeg_decode_frame(AVCodecContext *avctx, 

                              void *data, int *data_size,

                              UINT8 *buf, int buf_size)

{

    MJpegDecodeContext *s = avctx->priv_data;

    UINT8 *buf_end, *buf_ptr;

    int i, start_code;

    AVPicture *picture = data;



    *data_size = 0;



    /* no supplementary picture */

    if (buf_size == 0)

        return 0;



    buf_ptr = buf;

    buf_end = buf + buf_size;

    while (buf_ptr < buf_end) {

        /* find start next marker */

        start_code = find_marker(&buf_ptr, buf_end);

	{

	    /* EOF */

            if (start_code < 0) {

		goto the_end;

            } else {

                dprintf("marker=%x avail_size_in_buf=%d\n", start_code, buf_end - buf_ptr);

		

		if ((buf_end - buf_ptr) > s->buffer_size)

		{

		    av_free(s->buffer);

		    s->buffer_size = buf_end-buf_ptr;

		    s->buffer = av_malloc(s->buffer_size);

		    dprintf("buffer too small, expanding to %d bytes\n",

			s->buffer_size);

		}

		

		/* unescape buffer of SOS */

		if (start_code == SOS)

		{

		    UINT8 *src = buf_ptr;

		    UINT8 *dst = s->buffer;



		    while (src<buf_end)

		    {

			UINT8 x = *(src++);



			*(dst++) = x;

			if (x == 0xff)

			{

			    while(*src == 0xff) src++;



			    x = *(src++);

			    if (x >= 0xd0 && x <= 0xd7)

				*(dst++) = x;

			    else if (x)

				break;

			}

		    }

		    init_get_bits(&s->gb, s->buffer, dst - s->buffer);

		    

		    dprintf("escaping removed %d bytes\n",

			(buf_end - buf_ptr) - (dst - s->buffer));

		}

		else

		    init_get_bits(&s->gb, buf_ptr, buf_end - buf_ptr);

		

		s->start_code = start_code;



		/* process markers */

		if (start_code >= 0xd0 && start_code <= 0xd7) {

		    dprintf("restart marker: %d\n", start_code&0x0f);

		} else if (s->first_picture) {

		    /* APP fields */

		    if (start_code >= 0xe0 && start_code <= 0xef)

			mjpeg_decode_app(s);

		    /* Comment */

		    else if (start_code == COM)

			mjpeg_decode_com(s);

		}



                switch(start_code) {

                case SOI:

		    s->restart_interval = 0;

                    /* nothing to do on SOI */

                    break;

                case DQT:

                    mjpeg_decode_dqt(s);

                    break;

                case DHT:

                    mjpeg_decode_dht(s);

                    break;

                case SOF0:

                    if (mjpeg_decode_sof0(s) < 0)

			return -1;

                    break;

		case EOI:

eoi_parser:

		    {

                        if (s->interlaced) {

                            s->bottom_field ^= 1;

                            /* if not bottom field, do not output image yet */

                            if (s->bottom_field)

                                goto not_the_end;

                        }

                        for(i=0;i<3;i++) {

                            picture->data[i] = s->current_picture[i];

			    picture->linesize[i] = (s->interlaced) ?

				s->linesize[i] >> 1 : s->linesize[i];

                        }

                        *data_size = sizeof(AVPicture);

                        avctx->height = s->height;

                        if (s->interlaced)

                            avctx->height *= 2;

                        avctx->width = s->width;

                        /* XXX: not complete test ! */

                        switch((s->h_count[0] << 4) | s->v_count[0]) {

                        case 0x11:

                            avctx->pix_fmt = PIX_FMT_YUV444P;

                            break;

                        case 0x21:

                            avctx->pix_fmt = PIX_FMT_YUV422P;

                            break;

                        default:

                        case 0x22:

                            avctx->pix_fmt = PIX_FMT_YUV420P;

                            break;

                        }

                        /* dummy quality */

                        /* XXX: infer it with matrix */

//                    	avctx->quality = 3; 

                        goto the_end;

                    }

		    break;

                case SOS:

                    mjpeg_decode_sos(s);

		    /* buggy avid puts EOI every 10-20th frame */

		    /* if restart period is over process EOI */

		    if ((s->buggy_avid && !s->interlaced) || s->restart_interval)

			goto eoi_parser;

                    break;

		case DRI:

		    mjpeg_decode_dri(s);

		    break;

		case SOF1:

		case SOF2:

		case SOF3:

		case SOF5:

		case SOF6:

		case SOF7:

		case SOF9:

		case SOF10:

		case SOF11:

		case SOF13:

		case SOF14:

		case SOF15:

		case JPG:

		    printf("mjpeg: unsupported coding type (%x)\n", start_code);

		    break;

//		default:

//		    printf("mjpeg: unsupported marker (%x)\n", start_code);

//		    break;

                }



not_the_end:

		/* eof process start code */

		buf_ptr += (get_bits_count(&s->gb)+7)/8;

		dprintf("marker parser used %d bytes (%d bits)\n",

		    (get_bits_count(&s->gb)+7)/8, get_bits_count(&s->gb));

            }

        }

    }

the_end:

    dprintf("mjpeg decode frame unused %d bytes\n", buf_end - buf_ptr);

//    return buf_end - buf_ptr;

    return buf_ptr - buf;

}
