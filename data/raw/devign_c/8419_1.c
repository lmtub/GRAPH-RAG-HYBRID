static int cinepak_decode_strip (CinepakContext *s,
                                 cvid_strip_t *strip, uint8_t *data, int size)
{
    uint8_t *eod = (data + size);
    int      chunk_id, chunk_size;
    /* coordinate sanity checks */
    if (strip->x1 >= s->width  || strip->x2 > s->width  ||
        strip->y1 >= s->height || strip->y2 > s->height ||
        strip->x1 >= strip->x2 || strip->y1 >= strip->y2)
    while ((data + 4) <= eod) {
        chunk_id   = BE_16 (&data[0]);
        chunk_size = BE_16 (&data[2]) - 4;
        data      += 4;
        chunk_size = ((data + chunk_size) > eod) ? (eod - data) : chunk_size;
        switch (chunk_id) {
        case 0x2000:
        case 0x2100:
        case 0x2400:
        case 0x2500:
            cinepak_decode_codebook (strip->v4_codebook, chunk_id, 
                chunk_size, data);
            break;
        case 0x2200:
        case 0x2300:
        case 0x2600:
        case 0x2700:
            cinepak_decode_codebook (strip->v1_codebook, chunk_id, 
                chunk_size, data);
            break;
        case 0x3000:
        case 0x3100:
        case 0x3200:
            return cinepak_decode_vectors (s, strip, chunk_id, 
                chunk_size, data);
        }
        data += chunk_size;
    }
}