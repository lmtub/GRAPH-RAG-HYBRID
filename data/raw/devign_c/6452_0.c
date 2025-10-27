static void h261_v_loop_filter_c(uint8_t *dest,uint8_t *src, int stride){

    int i,j,xy,yz;

    int res;

    for(i=0; i<8; i++){

        for(j=1; j<7; j++){

            xy = j * stride + i;

            yz = j * 8 + i;

            res = (int)src[yz-1*8] + ((int)(src[yz+0*8]) * 2) + (int)src[yz+1*8];

            res +=2;

            res >>=2;

            dest[xy] = (uint8_t)res;

        }

    }

}
