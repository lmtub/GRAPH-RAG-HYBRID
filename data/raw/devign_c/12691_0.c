static int mov_read_default(MOVContext *c, ByteIOContext *pb, MOV_atom_t atom)

{

    int64_t total_size = 0;

    MOV_atom_t a;

    int i;

    int err = 0;



    a.offset = atom.offset;



    if (atom.size < 0)

        atom.size = INT64_MAX;

    while(((total_size + 8) < atom.size) && !url_feof(pb) && !err) {

        a.size = atom.size;

        a.type=0L;

        if(atom.size >= 8) {

            a.size = get_be32(pb);

            a.type = get_le32(pb);

        }

        total_size += 8;

        a.offset += 8;

        dprintf(c->fc, "type: %08x  %.4s  sz: %"PRIx64"  %"PRIx64"   %"PRIx64"\n", a.type, (char*)&a.type, a.size, atom.size, total_size);

        if (a.size == 1) { /* 64 bit extended size */

            a.size = get_be64(pb) - 8;

            a.offset += 8;

            total_size += 8;

        }

        if (a.size == 0) {

            a.size = atom.size - total_size;

            if (a.size <= 8)

                break;

        }

        a.size -= 8;

        if(a.size < 0 || a.size > atom.size - total_size)

            break;



        for (i = 0; c->parse_table[i].type != 0L

             && c->parse_table[i].type != a.type; i++)

            /* empty */;



        if (c->parse_table[i].type == 0) { /* skip leaf atoms data */

            url_fskip(pb, a.size);

        } else {

            offset_t start_pos = url_ftell(pb);

            int64_t left;

            err = (c->parse_table[i].func)(c, pb, a);

            left = a.size - url_ftell(pb) + start_pos;

            if (left > 0) /* skip garbage at atom end */

                url_fskip(pb, left);

        }



        a.offset += a.size;

        total_size += a.size;

    }



    if (!err && total_size < atom.size && atom.size < 0x7ffff) {

        url_fskip(pb, atom.size - total_size);

    }



    return err;

}
