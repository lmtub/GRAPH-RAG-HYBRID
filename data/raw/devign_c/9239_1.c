static int find_tag(ByteIOContext *pb, uint32_t tag1)

{

    unsigned int tag;

    int size;



    for(;;) {

        if (url_feof(pb))

            return -1;

        tag = get_le32(pb);

        size = get_le32(pb);

        if (tag == tag1)

            break;

        url_fseek(pb, size, SEEK_CUR);

    }

    if (size < 0)

        size = 0x7fffffff;

    return size;

}
