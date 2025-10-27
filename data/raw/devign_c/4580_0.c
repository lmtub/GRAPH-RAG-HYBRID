static int crc_write_header(struct AVFormatContext *s)

{

    CRCState *crc = s->priv_data;



    /* init CRC */

    crc->crcval = adler32(0, NULL, 0);



    return 0;

}
