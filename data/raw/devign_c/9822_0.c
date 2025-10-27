void ff_vp3_idct_dc_add_c(uint8_t *dest/*align 8*/, int line_size, const DCTELEM *block/*align 16*/){

    int i, dc = (block[0] + 15) >> 5;



    for(i = 0; i < 8; i++){

        dest[0] = av_clip_uint8(dest[0] + dc);

        dest[1] = av_clip_uint8(dest[1] + dc);

        dest[2] = av_clip_uint8(dest[2] + dc);

        dest[3] = av_clip_uint8(dest[3] + dc);

        dest[4] = av_clip_uint8(dest[4] + dc);

        dest[5] = av_clip_uint8(dest[5] + dc);

        dest[6] = av_clip_uint8(dest[6] + dc);

        dest[7] = av_clip_uint8(dest[7] + dc);

        dest += line_size;

    }

}
