matroska_probe (AVProbeData *p)

{

    uint64_t total = 0;

    int len_mask = 0x80, size = 1, n = 1;

    uint8_t probe_data[] = { 'm', 'a', 't', 'r', 'o', 's', 'k', 'a' };



    if (p->buf_size < 5)

        return 0;



    /* ebml header? */

    if ((p->buf[0] << 24 | p->buf[1] << 16 |

         p->buf[2] << 8 | p->buf[3]) != EBML_ID_HEADER)

        return 0;



    /* length of header */

    total = p->buf[4];

    while (size <= 8 && !(total & len_mask)) {

        size++;

        len_mask >>= 1;

    }

    if (size > 8)

      return 0;

    total &= (len_mask - 1);

    while (n < size)

        total = (total << 8) | p->buf[4 + n++];



    /* does the probe data contain the whole header? */

    if (p->buf_size < 4 + size + total)

      return 0;



    /* the header must contain the document type 'matroska'. For now,

     * we don't parse the whole header but simply check for the

     * availability of that array of characters inside the header.

     * Not fully fool-proof, but good enough. */

    for (n = 4 + size; n <= 4 + size + total - sizeof(probe_data); n++)

        if (!memcmp (&p->buf[n], probe_data, sizeof(probe_data)))

            return AVPROBE_SCORE_MAX;



    return 0;

}
