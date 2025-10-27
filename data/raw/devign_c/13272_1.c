static void pprint_data(V9fsPDU *pdu, int rx, size_t *offsetp, const char *name)

{

    struct iovec *sg = get_sg(pdu, rx);

    size_t offset = *offsetp;

    unsigned int count;

    int32_t size;

    int total, i, j;

    ssize_t len;



    if (rx) {

        count = pdu->elem.in_num;

    } else

        count = pdu->elem.out_num;

    }
