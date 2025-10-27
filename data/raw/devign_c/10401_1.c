static int mjpeg_decode_com(MJpegDecodeContext *s)

{

    /* XXX: verify len field validity */

    int len = get_bits(&s->gb, 16);

    if (len >= 2 && len < 32768) {

	/* XXX: any better upper bound */

	uint8_t *cbuf = av_malloc(len - 1);

	if (cbuf) {

	    int i;

	    for (i = 0; i < len - 2; i++)

		cbuf[i] = get_bits(&s->gb, 8);

	    if (i > 0 && cbuf[i-1] == '\n')

		cbuf[i-1] = 0;

	    else

		cbuf[i] = 0;



            if(s->avctx->debug & FF_DEBUG_PICT_INFO)

                av_log(s->avctx, AV_LOG_INFO, "mjpeg comment: '%s'\n", cbuf);



	    /* buggy avid, it puts EOI only at every 10th frame */

	    if (!strcmp(cbuf, "AVID"))

	    {

		s->buggy_avid = 1;

		//	if (s->first_picture)

		//	    printf("mjpeg: workarounding buggy AVID\n");

	    }

            else if(!strcmp(cbuf, "CS=ITU601")){

                s->cs_itu601= 1;

            }



	    av_free(cbuf);

	}

    }



    return 0;

}
