static int encode_codebook(CinepakEncContext *s, int *codebook, int size, int chunk_type_yuv, int chunk_type_gray, unsigned char *buf)

{

    int x, y, ret, entry_size = s->pix_fmt == AV_PIX_FMT_YUV420P ? 6 : 4;



    ret = write_chunk_header(buf, s->pix_fmt == AV_PIX_FMT_YUV420P ? chunk_type_yuv : chunk_type_gray, entry_size * size);



    for(x = 0; x < size; x++)

        for(y = 0; y < entry_size; y++)

            buf[ret++] = codebook[y + x*entry_size] ^ (y >= 4 ? 0x80 : 0);



    return ret;

}
