int h261_decode_picture_header(H261Context *h){

    MpegEncContext * const s = &h->s;

    int format, i;

    static int h261_framecounter = 0;

    uint32_t startcode;

    align_get_bits(&s->gb);



    startcode = (h->last_bits << (12 - (8-h->bits_left))) | get_bits(&s->gb, 20-8 - (8- h->bits_left));



    for(i= s->gb.size_in_bits - get_bits_count(&s->gb); i>24; i-=1){

        startcode = ((startcode << 1) | get_bits(&s->gb, 1)) & 0x000FFFFF;



        if(startcode == 0x10)

            break;

    }



    if (startcode != 0x10){

        av_log(s->avctx, AV_LOG_ERROR, "Bad picture start code\n");

        return -1;

    }



    /* temporal reference */

    s->picture_number = get_bits(&s->gb, 5); /* picture timestamp */



    /* PTYPE starts here */

    skip_bits1(&s->gb); /* split screen off */

    skip_bits1(&s->gb); /* camera  off */

    skip_bits1(&s->gb); /* freeze picture release off */



    format = get_bits1(&s->gb);



    //only 2 formats possible

    if (format == 0){//QCIF

        s->width = 176;

        s->height = 144;

        s->mb_width = 11;

        s->mb_height = 9;

    }else{//CIF

        s->width = 352;

        s->height = 288;

        s->mb_width = 22;

        s->mb_height = 18;

    }



    s->mb_num = s->mb_width * s->mb_height;



    skip_bits1(&s->gb); /* still image mode off */

    skip_bits1(&s->gb); /* Reserved */



    /* PEI */

    while (get_bits1(&s->gb) != 0){

        skip_bits(&s->gb, 8);

    }



    //h261 has no I-FRAMES, pass the test in MPV_frame_start in mpegvideo.c

    if(h261_framecounter > 1)

        s->pict_type = P_TYPE;

    else

        s->pict_type = I_TYPE;



    h261_framecounter++;



    h->gob_number = 0;

    return 0;

}
