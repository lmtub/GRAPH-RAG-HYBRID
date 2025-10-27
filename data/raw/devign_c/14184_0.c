void ff_build_rac_states(RangeCoder *c, int factor, int max_p){

    const int64_t one= 1LL<<32;

    int64_t p;

    int last_p8, p8, i;



    memset(c->zero_state, 0, sizeof(c->zero_state));

    memset(c-> one_state, 0, sizeof(c-> one_state));



#if 0

    for(i=1; i<256; i++){

        if(c->one_state[i])

            continue;



        p= (i*one + 128) >> 8;

        last_p8= i;

        for(;;){

            p+= ((one-p)*factor + one/2) >> 32;

            p8= (256*p + one/2) >> 32; //FIXME try without the one

            if(p8 <= last_p8) p8= last_p8+1;

            if(p8 > max_p) p8= max_p;

            if(p8 < last_p8)

                break;

            c->one_state[last_p8]=     p8;

            if(p8 == last_p8)

                break;

            last_p8= p8;

        }

    }

#endif

#if 1

    last_p8= 0;

    p= one/2;

    for(i=0; i<128; i++){

        p8= (256*p + one/2) >> 32; //FIXME try without the one

        if(p8 <= last_p8) p8= last_p8+1;

        if(last_p8 && last_p8<256 && p8<=max_p)

            c->one_state[last_p8]= p8;



        p+= ((one-p)*factor + one/2) >> 32;

        last_p8= p8;

    }

#endif

    for(i=256-max_p; i<=max_p; i++){

        if(c->one_state[i])

            continue;



        p= (i*one + 128) >> 8;

        p+= ((one-p)*factor + one/2) >> 32;

        p8= (256*p + one/2) >> 32; //FIXME try without the one

        if(p8 <= i) p8= i+1;

        if(p8 > max_p) p8= max_p;

        c->one_state[    i]=     p8;

    }



    for(i=0; i<256; i++)

        c->zero_state[i]= 256-c->one_state[256-i];

#if 0

    for(i=0; i<256; i++)

        av_log(NULL, AV_LOG_DEBUG, "%3d %3d\n", i, c->one_state[i]);

#endif

}
