static int dvvideo_decode_frame(AVCodecContext *avctx, 

                                 void *data, int *data_size,

                                 UINT8 *buf, int buf_size)

{

    DVVideoDecodeContext *s = avctx->priv_data;

    int sct, dsf, apt, ds, nb_dif_segs, vs, width, height, i, packet_size;

    unsigned size;

    UINT8 *buf_ptr;

    const UINT16 *mb_pos_ptr;

    AVPicture *picture;

    

    /* parse id */

    init_get_bits(&s->gb, buf, buf_size);

    sct = get_bits(&s->gb, 3);

    if (sct != 0)

        return -1;

    skip_bits(&s->gb, 5);

    get_bits(&s->gb, 4); /* dsn (sequence number */

    get_bits(&s->gb, 1); /* fsc (channel number) */

    skip_bits(&s->gb, 3);

    get_bits(&s->gb, 8); /* dbn (diff block number 0-134) */



    dsf = get_bits(&s->gb, 1); /* 0 = NTSC 1 = PAL */

    if (get_bits(&s->gb, 1) != 0)

        return -1;

    skip_bits(&s->gb, 11);

    apt = get_bits(&s->gb, 3); /* apt */



    get_bits(&s->gb, 1); /* tf1 */

    skip_bits(&s->gb, 4);

    get_bits(&s->gb, 3); /* ap1 */



    get_bits(&s->gb, 1); /* tf2 */

    skip_bits(&s->gb, 4);

    get_bits(&s->gb, 3); /* ap2 */



    get_bits(&s->gb, 1); /* tf3 */

    skip_bits(&s->gb, 4);

    get_bits(&s->gb, 3); /* ap3 */

    

    /* init size */

    width = 720;

    if (dsf) {

        avctx->frame_rate = 25 * FRAME_RATE_BASE;

        packet_size = PAL_FRAME_SIZE;

        height = 576;

        nb_dif_segs = 12;

    } else {

        avctx->frame_rate = 30 * FRAME_RATE_BASE;

        packet_size = NTSC_FRAME_SIZE;

        height = 480;

        nb_dif_segs = 10;

    }

    /* NOTE: we only accept several full frames */

    if (buf_size < packet_size)

        return -1;

    

    /* XXX: is it correct to assume that 420 is always used in PAL

       mode ? */

    s->sampling_411 = !dsf;

    if (s->sampling_411) {

        mb_pos_ptr = dv_place_411;

        avctx->pix_fmt = PIX_FMT_YUV411P;

    } else {

        mb_pos_ptr = dv_place_420;

        avctx->pix_fmt = PIX_FMT_YUV420P;

    }



    avctx->width = width;

    avctx->height = height;



    if (avctx->flags & CODEC_FLAG_DR1 && avctx->get_buffer_callback)

    {

	s->width = -1;

	avctx->dr_buffer[0] = avctx->dr_buffer[1] = avctx->dr_buffer[2] = 0;

	if(avctx->get_buffer_callback(avctx, width, height, I_TYPE) < 0){

	    fprintf(stderr, "get_buffer() failed\n");

	    return -1;

	}

    }



    /* (re)alloc picture if needed */

    if (s->width != width || s->height != height) {

	if (!(avctx->flags & CODEC_FLAG_DR1))

	    for(i=0;i<3;i++) {

		if (avctx->dr_buffer[i] != s->current_picture[i])

		    av_freep(&s->current_picture[i]);

		avctx->dr_buffer[i] = 0;

	    }



        for(i=0;i<3;i++) {

	    if (avctx->dr_buffer[i]) {

		s->current_picture[i] = avctx->dr_buffer[i];

		s->linesize[i] = (i == 0) ? avctx->dr_stride : avctx->dr_uvstride;

	    } else {

		size = width * height;

		s->linesize[i] = width;

		if (i >= 1) {

		    size >>= 2;

		    s->linesize[i] >>= s->sampling_411 ? 2 : 1;

		}

		s->current_picture[i] = av_malloc(size);

	    }

            if (!s->current_picture[i])

                return -1;

        }

        s->width = width;

        s->height = height;

    }



    /* for each DIF segment */

    buf_ptr = buf;

    for (ds = 0; ds < nb_dif_segs; ds++) {

        buf_ptr += 6 * 80; /* skip DIF segment header */

        

        for(vs = 0; vs < 27; vs++) {

            if ((vs % 3) == 0) {

                /* skip audio block */

                buf_ptr += 80;

            }

            dv_decode_video_segment(s, buf_ptr, mb_pos_ptr);

            buf_ptr += 5 * 80;

            mb_pos_ptr += 5;

        }

    }



    emms_c();



    /* return image */

    *data_size = sizeof(AVPicture);

    picture = data;

    for(i=0;i<3;i++) {

        picture->data[i] = s->current_picture[i];

        picture->linesize[i] = s->linesize[i];

    }

    return packet_size;

}
