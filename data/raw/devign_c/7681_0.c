static void h261_h_loop_filter_c(uint8_t *dest,uint8_t *src, int stride){

    int i,j,xy,yz;

    int res;    

    for(i=1; i<7; i++){

        for(j=0; j<8; j++){

            xy = j * stride + i;

            yz = j * 8 + i;

            res = (int)src[yz-1] + ((int)(src[yz]) *2) + (int)src[yz+1];

            res+=2;

            res>>=2;

            dest[xy] = (uint8_t)res;

        }

    }

}
