static int mov_read_wide(MOVContext *c, ByteIOContext *pb, MOV_atom_t atom)

{

    int err;

    uint32_t type;



#ifdef DEBUG

    print_atom("wide", atom);

    debug_indent++;

#endif

    if (atom.size < 8)

        return 0; /* continue */

    if (get_be32(pb) != 0) { /* 0 sized mdat atom... use the 'wide' atom size */

        url_fskip(pb, atom.size - 4);

        return 0;

    }

    atom.type = get_le32(pb);

    atom.offset += 8;

    atom.size -= 8;

    if (type != MKTAG('m', 'd', 'a', 't')) {

        url_fskip(pb, atom.size);

        return 0;

    }

    err = mov_read_mdat(c, pb, atom);

#ifdef DEBUG

    debug_indent--;

#endif

    return err;

}
