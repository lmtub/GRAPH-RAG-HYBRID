static int mxf_read_generic_descriptor(MXFDescriptor *descriptor, ByteIOContext *pb, int tag, int size, UID uid)

{

    switch(tag) {

    case 0x3F01:

        descriptor->sub_descriptors_count = get_be32(pb);

        if (descriptor->sub_descriptors_count >= UINT_MAX / sizeof(UID))

            return -1;

        descriptor->sub_descriptors_refs = av_malloc(descriptor->sub_descriptors_count * sizeof(UID));

        if (!descriptor->sub_descriptors_refs)

            return -1;

        url_fskip(pb, 4); /* useless size of objects, always 16 according to specs */

        get_buffer(pb, (uint8_t *)descriptor->sub_descriptors_refs, descriptor->sub_descriptors_count * sizeof(UID));

        break;

    case 0x3004:

        get_buffer(pb, descriptor->essence_container_ul, 16);

        break;

    case 0x3006:

        descriptor->linked_track_id = get_be32(pb);

        break;

    case 0x3201: /* PictureEssenceCoding */

        get_buffer(pb, descriptor->essence_codec_ul, 16);

        break;

    case 0x3203:

        descriptor->width = get_be32(pb);

        break;

    case 0x3202:

        descriptor->height = get_be32(pb);

        break;

    case 0x320E:

        descriptor->aspect_ratio.num = get_be32(pb);

        descriptor->aspect_ratio.den = get_be32(pb);

        break;

    case 0x3D03:

        descriptor->sample_rate.num = get_be32(pb);

        descriptor->sample_rate.den = get_be32(pb);

        break;

    case 0x3D06: /* SoundEssenceCompression */

        get_buffer(pb, descriptor->essence_codec_ul, 16);

        break;

    case 0x3D07:

        descriptor->channels = get_be32(pb);

        break;

    case 0x3D01:

        descriptor->bits_per_sample = get_be32(pb);

        break;

    case 0x3401:

        mxf_read_pixel_layout(pb, descriptor);

        break;

    default:

        /* Private uid used by SONY C0023S01.mxf */

        if (IS_KLV_KEY(uid, mxf_sony_mpeg4_extradata)) {

            descriptor->extradata = av_malloc(size);

            if (!descriptor->extradata)

                return -1;

            descriptor->extradata_size = size;

            get_buffer(pb, descriptor->extradata, size);

        }

        break;

    }

    return 0;

}
