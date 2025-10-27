static int crc_write_packet(struct AVFormatContext *s, 

                            int stream_index,

                            const uint8_t *buf, int size, int64_t pts)

{

    CRCState *crc = s->priv_data;

    crc->crcval = adler32(crc->crcval, buf, size);

    return 0;

}
