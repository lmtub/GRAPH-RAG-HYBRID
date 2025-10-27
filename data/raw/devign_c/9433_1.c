static int read_rle_sgi(unsigned char* out_buf, const uint8_t *in_buf,

                        const uint8_t *in_end, SgiState* s)

{

    uint8_t *dest_row;

    unsigned int len = s->height * s->depth * 4;

    const uint8_t *start_table = in_buf;

    unsigned int y, z;

    unsigned int start_offset;



    /* size of  RLE offset and length tables */

    if(len * 2  > in_end - in_buf) {

        return AVERROR_INVALIDDATA;

    }



    in_buf -= SGI_HEADER_SIZE;

    for (z = 0; z < s->depth; z++) {

        dest_row = out_buf;

        for (y = 0; y < s->height; y++) {

            dest_row -= s->linesize;

            start_offset = bytestream_get_be32(&start_table);

            if(start_offset > in_end - in_buf) {

                return AVERROR_INVALIDDATA;

            }

            if (expand_rle_row(in_buf + start_offset, in_end, dest_row + z,

                dest_row + FFABS(s->linesize), s->depth) != s->width)

                return AVERROR_INVALIDDATA;

        }

    }

    return 0;

}
