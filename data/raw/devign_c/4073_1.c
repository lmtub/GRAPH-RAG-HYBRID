static int mxf_read_sequence(void *arg, AVIOContext *pb, int tag, int size, UID uid)

{

    MXFSequence *sequence = arg;

    switch(tag) {

    case 0x0202:

        sequence->duration = avio_rb64(pb);

        break;

    case 0x0201:

        avio_read(pb, sequence->data_definition_ul, 16);

        break;

    case 0x1001:

        sequence->structural_components_count = avio_rb32(pb);

        if (sequence->structural_components_count >= UINT_MAX / sizeof(UID))

            return -1;

        sequence->structural_components_refs = av_malloc(sequence->structural_components_count * sizeof(UID));

        if (!sequence->structural_components_refs)

            return -1;

        avio_skip(pb, 4); /* useless size of objects, always 16 according to specs */

        avio_read(pb, (uint8_t *)sequence->structural_components_refs, sequence->structural_components_count * sizeof(UID));

        break;

    }

    return 0;

}
