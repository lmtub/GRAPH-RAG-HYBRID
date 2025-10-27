static int add_metadata(const uint8_t **buf, int count, int type,

                        const char *name, const char *sep, TiffContext *s)

{

    switch(type) {

    case TIFF_DOUBLE: return add_doubles_metadata(buf, count, name, sep, s);

    case TIFF_SHORT : return add_shorts_metadata(buf, count, name, sep, s);

    default         : return AVERROR_INVALIDDATA;

    };

}
