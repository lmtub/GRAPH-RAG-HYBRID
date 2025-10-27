static unsigned int codec_get_asf_tag(const CodecTag *tags, unsigned int id)

{

    while (tags->id != 0) {

        if (!tags->invalid_asf && tags->id == id)

            return tags->tag;

        tags++;

    }

    return 0;

}
