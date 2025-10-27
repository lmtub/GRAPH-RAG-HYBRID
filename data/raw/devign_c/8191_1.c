static int mxf_read_sequence(MXFSequence *sequence, ByteIOContext *pb, int tag)

{

    switch(tag) {

    case 0x0202:

        sequence->duration = get_be64(pb);

        break;

    case 0x0201:

        get_buffer(pb, sequence->data_definition_ul, 16);

        break;

    case 0x1001:

        sequence->structural_components_count = get_be32(pb);

        if (sequence->structural_components_count >= UINT_MAX / sizeof(UID))

            return -1;

        sequence->structural_components_refs = av_malloc(sequence->structural_components_count * sizeof(UID));

        if (!sequence->structural_components_refs)

            return -1;

        url_fskip(pb, 4); /* useless size of objects, always 16 according to specs */

        get_buffer(pb, (uint8_t *)sequence->structural_components_refs, sequence->structural_components_count * sizeof(UID));

        break;

    }

    return 0;

}
