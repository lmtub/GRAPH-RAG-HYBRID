const uint8_t *ff_h264_decode_nal(H264Context *h, const uint8_t *src, int *dst_length, int *consumed, int length){

    int i, si, di;

    uint8_t *dst;

    int bufidx;



//    src[0]&0x80;                //forbidden bit

    h->nal_ref_idc= src[0]>>5;

    h->nal_unit_type= src[0]&0x1F;



    src++; length--;



#if HAVE_FAST_UNALIGNED

# if HAVE_FAST_64BIT

#   define RS 7

    for(i=0; i+1<length; i+=9){

        if(!((~AV_RN64A(src+i) & (AV_RN64A(src+i) - 0x0100010001000101ULL)) & 0x8000800080008080ULL))

# else

#   define RS 3

    for(i=0; i+1<length; i+=5){

        if(!((~AV_RN32A(src+i) & (AV_RN32A(src+i) - 0x01000101U)) & 0x80008080U))

# endif

            continue;

        if(i>0 && !src[i]) i--;

        while(src[i]) i++;

#else

#   define RS 0

    for(i=0; i+1<length; i+=2){

        if(src[i]) continue;

        if(i>0 && src[i-1]==0) i--;

#endif

        if(i+2<length && src[i+1]==0 && src[i+2]<=3){

            if(src[i+2]!=3){

                /* startcode, so we must be past the end */

                length=i;

            }

            break;

        }

        i-= RS;

    }



    if(i>=length-1){ //no escaped 0

        *dst_length= length;

        *consumed= length+1; //+1 for the header

        return src;

    }



    bufidx = h->nal_unit_type == NAL_DPC ? 1 : 0; // use second escape buffer for inter data

    av_fast_malloc(&h->rbsp_buffer[bufidx], &h->rbsp_buffer_size[bufidx], length+FF_INPUT_BUFFER_PADDING_SIZE);

    dst= h->rbsp_buffer[bufidx];



    if (dst == NULL){

        return NULL;

    }



//printf("decoding esc\n");

    memcpy(dst, src, i);

    si=di=i;

    while(si+2<length){

        //remove escapes (very rare 1:2^22)

        if(src[si+2]>3){

            dst[di++]= src[si++];

            dst[di++]= src[si++];

        }else if(src[si]==0 && src[si+1]==0){

            if(src[si+2]==3){ //escape

                dst[di++]= 0;

                dst[di++]= 0;

                si+=3;

                continue;

            }else //next start code

                goto nsc;

        }



        dst[di++]= src[si++];

    }

    while(si<length)

        dst[di++]= src[si++];

nsc:



    memset(dst+di, 0, FF_INPUT_BUFFER_PADDING_SIZE);



    *dst_length= di;

    *consumed= si + 1;//+1 for the header

//FIXME store exact number of bits in the getbitcontext (it is needed for decoding)

    return dst;

}
