static void init_dequant8_coeff_table(H264Context *h){

    int i,q,x;

    const int transpose = (h->h264dsp.h264_idct8_add != ff_h264_idct8_add_c); //FIXME ugly

    h->dequant8_coeff[0] = h->dequant8_buffer[0];

    h->dequant8_coeff[1] = h->dequant8_buffer[1];



    for(i=0; i<2; i++ ){

        if(i && !memcmp(h->pps.scaling_matrix8[0], h->pps.scaling_matrix8[1], 64*sizeof(uint8_t))){

            h->dequant8_coeff[1] = h->dequant8_buffer[0];

            break;

        }



        for(q=0; q<52; q++){

            int shift = div6[q];

            int idx = rem6[q];

            for(x=0; x<64; x++)

                h->dequant8_coeff[i][q][transpose ? (x>>3)|((x&7)<<3) : x] =

                    ((uint32_t)dequant8_coeff_init[idx][ dequant8_coeff_init_scan[((x>>1)&12) | (x&3)] ] *

                    h->pps.scaling_matrix8[i][x]) << shift;

        }

    }

}
