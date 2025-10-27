static int jp2_find_codestream(Jpeg2000DecoderContext *s)

{

    uint32_t atom_size, atom, atom_end;

    int search_range = 10;



    while (search_range

           &&

           bytestream2_get_bytes_left(&s->g) >= 8) {

        atom_size = bytestream2_get_be32u(&s->g);

        atom      = bytestream2_get_be32u(&s->g);

        atom_end  = bytestream2_tell(&s->g) + atom_size - 8;



        if (atom == JP2_CODESTREAM)

            return 1;



        if (bytestream2_get_bytes_left(&s->g) < atom_size || atom_end < atom_size)

            return 0;



        if (atom == JP2_HEADER &&

                   atom_size >= 16) {

            uint32_t atom2_size, atom2, atom2_end;

            do {

                atom2_size = bytestream2_get_be32u(&s->g);

                atom2      = bytestream2_get_be32u(&s->g);

                atom2_end  = bytestream2_tell(&s->g) + atom2_size - 8;

                if (atom2_size < 8 || atom2_end > atom_end || atom2_end < atom2_size)

                    break;

                atom2_size -= 8;

                if (atom2 == JP2_CODESTREAM) {

                    return 1;

                } else if (atom2 == MKBETAG('c','o','l','r') && atom2_size >= 7) {

                    int method = bytestream2_get_byteu(&s->g);

                    bytestream2_skipu(&s->g, 2);

                    if (method == 1) {

                        s->colour_space = bytestream2_get_be32u(&s->g);


                } else if (atom2 == MKBETAG('p','c','l','r') && atom2_size >= 6) {

                    int i, size, colour_count, colour_channels, colour_depth[3];

                    uint32_t r, g, b;

                    colour_count = bytestream2_get_be16u(&s->g);

                    colour_channels = bytestream2_get_byteu(&s->g);

                    // FIXME: Do not ignore channel_sign

                    colour_depth[0] = (bytestream2_get_byteu(&s->g) & 0x7f) + 1;

                    colour_depth[1] = (bytestream2_get_byteu(&s->g) & 0x7f) + 1;

                    colour_depth[2] = (bytestream2_get_byteu(&s->g) & 0x7f) + 1;

                    size = (colour_depth[0] + 7 >> 3) * colour_count +

                           (colour_depth[1] + 7 >> 3) * colour_count +

                           (colour_depth[2] + 7 >> 3) * colour_count;

                    if (colour_count > 256   ||

                        colour_channels != 3 ||

                        colour_depth[0] > 16 ||

                        colour_depth[1] > 16 ||

                        colour_depth[2] > 16 ||

                        atom2_size < size) {

                        avpriv_request_sample(s->avctx, "Unknown palette");




                    s->pal8 = 1;

                    for (i = 0; i < colour_count; i++) {

                        if (colour_depth[0] <= 8) {

                            r = bytestream2_get_byteu(&s->g) << 8 - colour_depth[0];

                            r |= r >> colour_depth[0];

                        } else {

                            r = bytestream2_get_be16u(&s->g) >> colour_depth[0] - 8;


                        if (colour_depth[1] <= 8) {

                            g = bytestream2_get_byteu(&s->g) << 8 - colour_depth[1];

                            r |= r >> colour_depth[1];

                        } else {

                            g = bytestream2_get_be16u(&s->g) >> colour_depth[1] - 8;


                        if (colour_depth[2] <= 8) {

                            b = bytestream2_get_byteu(&s->g) << 8 - colour_depth[2];

                            r |= r >> colour_depth[2];

                        } else {

                            b = bytestream2_get_be16u(&s->g) >> colour_depth[2] - 8;


                        s->palette[i] = 0xffu << 24 | r << 16 | g << 8 | b;


                } else if (atom2 == MKBETAG('c','d','e','f') && atom2_size >= 2) {

                    int n = bytestream2_get_be16u(&s->g);

                    for (; n>0; n--) {

                        int cn   = bytestream2_get_be16(&s->g);

                        int av_unused typ  = bytestream2_get_be16(&s->g);

                        int asoc = bytestream2_get_be16(&s->g);

                        if (cn < 4 && asoc < 4)

                            s->cdef[cn] = asoc;


                } else if (atom2 == MKBETAG('r','e','s',' ') && atom2_size >= 18) {

                    int64_t vnum, vden, hnum, hden, vexp, hexp;

                    uint32_t resx;

                    bytestream2_skip(&s->g, 4);

                    resx = bytestream2_get_be32u(&s->g);

                    if (resx != MKBETAG('r','e','s','c') && resx != MKBETAG('r','e','s','d')) {




                    vnum = bytestream2_get_be16u(&s->g);

                    vden = bytestream2_get_be16u(&s->g);

                    hnum = bytestream2_get_be16u(&s->g);

                    hden = bytestream2_get_be16u(&s->g);

                    vexp = bytestream2_get_byteu(&s->g);

                    hexp = bytestream2_get_byteu(&s->g);






                    if (vexp > hexp) {

                        vexp -= hexp;

                        hexp = 0;

                    } else {

                        hexp -= vexp;

                        vexp = 0;


                    if (   INT64_MAX / (hnum * vden) > pow(10, hexp)

                        && INT64_MAX / (vnum * hden) > pow(10, vexp))

                        av_reduce(&s->sar.den, &s->sar.num,

                                  hnum * vden * pow(10, hexp),

                                  vnum * hden * pow(10, vexp),

                                  INT32_MAX);



            } while (atom_end - atom2_end >= 8);

        } else {

            search_range--;


        bytestream2_seek(&s->g, atom_end, SEEK_SET);




    return 0;
