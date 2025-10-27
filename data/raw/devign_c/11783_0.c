static void h261_encode_motion(H261Context * h, int val){

    MpegEncContext * const s = &h->s;

    int sign, code;

    if(val==0){

        code = 0;

        put_bits(&s->pb,h261_mv_tab[code][1],h261_mv_tab[code][0]);

    } 

    else{

        if(val > 16)

            val -=32;

        if(val < -16)

            val+=32;

        sign = val < 0;

        code = sign ? -val : val; 

        put_bits(&s->pb,h261_mv_tab[code][1],h261_mv_tab[code][0]);

        put_bits(&s->pb,1,sign);

    }

}
