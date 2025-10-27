static int decode_i2_frame(FourXContext *f, const uint8_t *buf, int length){

    int x, y, x2, y2;

    const int width= f->avctx->width;

    const int height= f->avctx->height;

    uint16_t *dst= (uint16_t*)f->current_picture.data[0];

    const int stride= f->current_picture.linesize[0]>>1;



    for(y=0; y<height; y+=16){

        for(x=0; x<width; x+=16){

            unsigned int color[4], bits;

            memset(color, 0, sizeof(color));

//warning following is purely guessed ...

            color[0]= bytestream_get_le16(&buf);

            color[1]= bytestream_get_le16(&buf);



            if(color[0]&0x8000) av_log(NULL, AV_LOG_ERROR, "unk bit 1\n");

            if(color[1]&0x8000) av_log(NULL, AV_LOG_ERROR, "unk bit 2\n");



            color[2]= mix(color[0], color[1]);

            color[3]= mix(color[1], color[0]);



            bits= bytestream_get_le32(&buf);

            for(y2=0; y2<16; y2++){

                for(x2=0; x2<16; x2++){

                    int index= 2*(x2>>2) + 8*(y2>>2);

                    dst[y2*stride+x2]= color[(bits>>index)&3];

                }

            }

            dst+=16;

        }

        dst += 16 * stride - x;

    }



    return 0;

}
