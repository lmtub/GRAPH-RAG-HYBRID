static int load_xbzrle(QEMUFile *f, ram_addr_t addr, void *host)

{

    int ret, rc = 0;

    unsigned int xh_len;

    int xh_flags;



    if (!XBZRLE.decoded_buf) {

        XBZRLE.decoded_buf = g_malloc(TARGET_PAGE_SIZE);

    }



    /* extract RLE header */

    xh_flags = qemu_get_byte(f);

    xh_len = qemu_get_be16(f);



    if (xh_flags != ENCODING_FLAG_XBZRLE) {

        fprintf(stderr, "Failed to load XBZRLE page - wrong compression!\n");

        return -1;

    }



    if (xh_len > TARGET_PAGE_SIZE) {

        fprintf(stderr, "Failed to load XBZRLE page - len overflow!\n");

        return -1;

    }

    /* load data and decode */

    qemu_get_buffer(f, XBZRLE.decoded_buf, xh_len);



    /* decode RLE */

    ret = xbzrle_decode_buffer(XBZRLE.decoded_buf, xh_len, host,

                               TARGET_PAGE_SIZE);

    if (ret == -1) {

        fprintf(stderr, "Failed to load XBZRLE page - decode error!\n");

        rc = -1;

    } else  if (ret > TARGET_PAGE_SIZE) {

        fprintf(stderr, "Failed to load XBZRLE page - size %d exceeds %d!\n",

                ret, TARGET_PAGE_SIZE);

        abort();

    }



    return rc;

}
