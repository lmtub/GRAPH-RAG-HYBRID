unsigned int codec_get_tag(const CodecTag *tags, int id)

{

    while (tags->id != 0) {

        if (tags->id == id)

            return tags->tag;

        tags++;

    }

    return 0;

}
